import pytest
import jax.numpy as jnp
from jax import jit
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dmff import Hamiltonian, NeighborList
from dmff.common.nblist import NeighborListFreud
from dmff.utils import regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import (
    distribute_v3, 
)
from dmff.admp.spatial import (
    v_pbc_shift, 
)
import freud

class TestNeighborList:
    
    @pytest.fixture(scope="class", name='nblist')
    def test_nblist_init(self):

        """load generators from XML file

        Yields:
            Tuple: (
                ADMPDispForce,
                ADMPPmeForce, # polarized
            )
        """
        rc = 4.0
        H = Hamiltonian('tests/data/admp.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        potential = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        generators = H.getGenerators()
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = np.array([a._value, b._value, c._value]) * 10
        positions = np.array(pdb.positions._value) * 10


        nbobj = NeighborList(box, rc, potential.meta["cov_map"])
        nbobj.allocate(positions)
        yield nbobj


    def test_update(self, nblist):

        positions = jnp.array([
            [12.434,   3.404,   1.540],
            [13.030,   2.664,   1.322],
            [12.312,   3.814,   0.660],
            [14.216,   1.424,   1.103],
            [14.246,   1.144,   2.054],
            [15.155,   1.542,   0.910]
        ])   
        nblist.update(positions)
        
    def test_pairs(self, nblist):
        
        pairs = nblist.pairs
        assert pairs.shape[0] == int(15 * 1.3)
        
    def test_scaled_pairs(self, nblist):
        scaled = pair_buffer_scales(nblist.pairs)
        assert scaled.sum() == 15


class TestFreudNeighborlist:

    @pytest.fixture(scope="class", name='nblist')
    def test_nblist_init(self):

        """load generators from XML file

        Yields:
            Tuple: (
                ADMPDispForce,
                ADMPPmeForce, # polarized
            )
        """
        rc = 4.0
        H = Hamiltonian('tests/data/admp.xml')
        pdb = app.PDBFile('tests/data/water_dimer.pdb')
        potential = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom, ethresh=5e-4, step_pol=5)
        a, b, c = pdb.topology.getPeriodicBoxVectors()
        box = np.array([a._value, b._value, c._value]) * 10
        positions = np.array(pdb.positions._value) * 10
        nbobj = NeighborListFreud(box, rc, potential.meta["cov_map"])
        nbobj.capacity_multiplier = 1
        nbobj.allocate(positions)
        yield nbobj

    def test_update(self, nblist):

        positions = jnp.array([
            [12.434,   3.404,   1.540],
            [13.030,   2.664,   1.322],
            [12.312,   3.814,   0.660],
            [14.216,   1.424,   1.103],
            [14.246,   1.144,   2.054],
            [15.155,   1.542,   0.910]
        ])   
        nblist.update(positions)
        
    def test_pairs(self, nblist):
        
        pairs = nblist.pairs
        assert pairs.shape == (15, 3)
        
    def test_scaled_pairs(self, nblist):
        
        pairs = nblist.scaled_pairs
        scaled = pair_buffer_scales(nblist.pairs)
        assert pairs.shape[0] == scaled.sum()


class TestBuildPairsBatch:
    """build_pairs_batch must reproduce the per-frame NeighborListFreud result,
    regardless of the number of worker processes."""

    @pytest.fixture(scope="class")
    def frames(self):
        from dmff.common.nblist import build_pairs_batch  # noqa: F401
        rng = np.random.default_rng(0)
        n_frames, n_atoms = 7, 60
        box = np.diag([2.0, 2.0, 2.0])
        coords = rng.uniform(0.0, 2.0, size=(n_frames, n_atoms, 3)).astype(np.float32)
        boxes = np.repeat(box[None], n_frames, axis=0)
        # a random-ish "covalent" map with a few non-zero entries
        cov_map = np.zeros((n_atoms, n_atoms), dtype=int)
        for i in range(0, n_atoms - 1, 3):
            cov_map[i, i + 1] = cov_map[i + 1, i] = 1
        return coords, boxes, cov_map

    def _reference(self, coords, boxes, cov_map, rcut):
        ref = []
        for c, b in zip(coords, boxes):
            nb = NeighborListFreud(b, rcut, cov_map)
            nb.capacity_multiplier = 1
            ref.append(np.asarray(nb.allocate(c)))
        pmax = max(p.shape[0] for p in ref)
        out = np.full((len(ref), pmax, 3), coords.shape[1], dtype=int)
        for i, p in enumerate(ref):
            out[i, : p.shape[0]] = p
        return out

    @pytest.mark.parametrize("n_workers", [1, 2, 3])
    def test_matches_per_frame(self, frames, n_workers):
        from dmff.common.nblist import build_pairs_batch
        coords, boxes, cov_map = frames
        rcut = 0.6
        ref = self._reference(coords, boxes, cov_map, rcut)
        got = build_pairs_batch(
            coords, boxes, rcut, cov_map, n_workers=n_workers, show_progress=False
        )
        assert got.shape == ref.shape
        # pair order inside a frame is not guaranteed -> compare as sets of rows
        for f in range(coords.shape[0]):
            r = ref[f][np.lexsort(ref[f].T[::-1])]
            g = got[f][np.lexsort(got[f].T[::-1])]
            npt.assert_array_equal(g, r)

    def test_single_box_broadcast(self, frames):
        from dmff.common.nblist import build_pairs_batch
        coords, boxes, cov_map = frames
        a = build_pairs_batch(coords, boxes, 0.6, cov_map, n_workers=1, show_progress=False)
        b = build_pairs_batch(coords, boxes[0], 0.6, cov_map, n_workers=1, show_progress=False)
        npt.assert_array_equal(a, b)

    def test_padding_masked(self, frames):
        from dmff.common.nblist import build_pairs_batch
        coords, boxes, cov_map = frames
        got = build_pairs_batch(coords, boxes, 0.6, cov_map, n_workers=2, show_progress=False)
        for f in range(coords.shape[0]):
            scales = np.asarray(pair_buffer_scales(jnp.array(got[f])))
            n_real = int((got[f][:, 0] < coords.shape[1]).sum())
            assert scales.sum() == n_real
            # padded rows carry the dummy index in every column
            assert (got[f][n_real:] == coords.shape[1]).all()
