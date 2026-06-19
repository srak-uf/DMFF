import numpy as np
import jax.numpy as jnp
from itertools import permutations
try:
    import freud
except ImportError:
    freud = None
    import warnings
    warnings.warn("WARNING: freud not installed, users need to create neighbor list by themselves.")
try:
    from ase import Atoms
except ImportError:
    Atoms = None
    import warnings
    warnings.warn("WARNING: ase not installed, users need to create neighbor list by themselves.")
try:
    import neighborlist_rs
except ImportError:
    neighborlist_rs = None
    import warnings
    warnings.warn("WARNING: neighborlist_rs not installed, users need to create neighbor list by themselves.")

class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True):
        if freud is None:
            raise ImportError("Freud not installed.")
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        aq = freud.locality.AABBQuery(fbox, coords)
        res = aq.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        nlist = res.toNeighborList()
        nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
        nlist = nlist.astype(np.int32)
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        if self.capacity_multiplier is None:
            self.capacity_multiplier = int(nlist.shape[0] * 1.3)
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions

class NeighborListRS:
    def __init__(self, box, rcut, cov_map, padding=True):
        if neighborlist_rs is None:
            raise ImportError("neighborlist_rs not installed.")
        self.box = np.asarray(box, dtype=np.float64)  # 初期 box（使わなくてもOK）
        self.rcut = float(rcut)                       # nm
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map

        self._pairs = None
        self._positions = None
        self._shifts = None
        self.nlist = None

    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        return jnp.concatenate([pairs, nbond[:, None]], axis=1)

    def _atoms_from_mdtraj(self, coords_nm, cell_nm):
        coords = np.asarray(coords_nm * 10.0, dtype=np.float64)   # nm → Å
        cell = np.asarray(cell_nm * 10.0, dtype=np.float64)       # nm → Å

        n = coords.shape[0]
        symbols = ["H"] * n  # 種類は cov_map 側で持っているので何でもよい

        atoms = Atoms(symbols=symbols, positions=coords, cell=cell, pbc=True)
        return atoms

    def allocate(self, coords, box=None):
        self._positions = coords

        coords_nm = np.asarray(coords, dtype=np.float64)
        cell_nm = self.box if box is None else np.asarray(box, dtype=np.float64)

        atoms = self._atoms_from_mdtraj(coords_nm, cell_nm)

        # cutoff も Å に変換
        result = neighborlist_rs.build_from_ase(atoms, cutoff=self.rcut * 10.0)

        edge_index = np.asarray(result["edge_index"], dtype=np.int32)   # (2, M)
        shifts = np.asarray(result["shift"], dtype=np.int32)            # (M, 3)

        nlist = edge_index.T
        nlist = np.vstack((nlist, nlist[:, ::-1]))
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        self.nlist = nlist
        if self.capacity_multiplier is None:
            self.capacity_multiplier = int(nlist.shape[0] * 1.3)
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        return self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions

    @property
    def shifts(self):
        return self._shifts


class NeighborList(NeighborListFreud):
    ...

class NoCutoffNeighborList:
    
    def __init__(self, cov_map, padding=True):
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        natoms = coords.shape[0]
        # nblist = np.fromiter(permutations(range(natoms), 2), dtype=np.dtype(int, 2))
        nblist = np.array(list(permutations(range(natoms), 2)), dtype=np.dtype(int, 2))
        nlist = nblist[nblist[:, 0] < nblist[:, 1]]
        if self.capacity_multiplier is None:
            self.capacity_multiplier = int(nlist.shape[0] * 1.3)
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        self.allocate(positions)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions


class NoPeriodicNeighborList(NoCutoffNeighborList):
    
    def __init__(self, rcut, cov_map, padding=True):
        super().__init__(cov_map, padding)
        self.rcut = rcut

    def allocate(self, coords):
        self._positions = coords  # cache it
        natoms = coords.shape[0]
        # nblist = np.fromiter(permutations(range(natoms), 2), dtype=np.dtype(int, 2))
        nblist = np.array(list(permutations(range(natoms), 2)), dtype=np.dtype(int, 2))
        nlist = nblist[nblist[:, 0] < nblist[:, 1]]
        distances = np.linalg.norm(coords[nlist[:, 0]] - coords[nlist[:, 1]], axis=1)
        nlist = nlist[distances < self.rcut]
        if self.capacity_multiplier is None:
            self.capacity_multiplier = int(nlist.shape[0] * 1.3)
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")