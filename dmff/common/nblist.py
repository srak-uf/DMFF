import os
import warnings
import functools
import contextlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
import numpy as np
import jax.numpy as jnp
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

# ---------------------------------------------------------------------------
# Low level (picklable, jax-free) pair builders
# ---------------------------------------------------------------------------
def _freud_raw_pairs(coords, box, rcut):
    """Return the (M, 2) int32 array of atom pairs (i < j) within ``rcut``
    for a single frame, using freud.

    ``coords`` and ``box`` are plain numpy arrays. No cov_map lookup and no
    padding is done here, so this function can be sent to worker processes.
    """
    if freud is None:
        raise ImportError("Freud not installed.")
    fbox = freud.box.Box.from_matrix(np.asarray(box))
    coords = np.asarray(coords)
    aq = freud.locality.AABBQuery(fbox, coords)
    res = aq.query(coords, dict(r_max=rcut, exclude_ii=True))
    nlist = res.toNeighborList()
    nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
    nlist = nlist.astype(np.int32)
    msk = (nlist[:, 0] - nlist[:, 1]) < 0
    return nlist[msk]


def _rs_raw_pairs(coords, box, rcut):
    """Return the (M, 2) int32 array of atom pairs (i < j) within ``rcut``
    for a single frame, using neighborlist_rs (input in nm, converted to Å).
    """
    if neighborlist_rs is None:
        raise ImportError("neighborlist_rs not installed.")
    if Atoms is None:
        raise ImportError("ase not installed.")
    coords = np.asarray(coords, dtype=np.float64) * 10.0   # nm → Å
    cell = np.asarray(box, dtype=np.float64) * 10.0        # nm → Å
    n = coords.shape[0]
    # 元素種は cov_map 側で持っているので何でもよい
    atoms = Atoms(symbols=["H"] * n, positions=coords, cell=cell, pbc=True)
    result = neighborlist_rs.build_from_ase(atoms, cutoff=float(rcut) * 10.0)
    edge_index = np.asarray(result["edge_index"], dtype=np.int32)   # (2, M)
    nlist = edge_index.T
    nlist = np.vstack((nlist, nlist[:, ::-1]))
    msk = (nlist[:, 0] - nlist[:, 1]) < 0
    return nlist[msk]


_RAW_PAIR_BUILDERS = {"freud": _freud_raw_pairs, "rs": _rs_raw_pairs}


def _set_freud_threads(n):
    # worker initializer: freud parallelises with TBB internally, so cap its
    # threads to avoid oversubscription when many workers run at once.
    if freud is not None:
        freud.parallel.set_num_threads(int(n))


def build_pairs_batch(
    coords_list,
    box_list,
    rcut,
    cov_map,
    backend="freud",
    n_workers=None,
    chunksize=None,
    pad_value=None,
    show_progress=True,
    mp_context=None,
    dtype=int,
):
    """Build the neighbor lists of many frames at once, distributing the
    frames over ``n_workers`` processes (``None``: all CPUs, ``1``: serial).

    ``coords_list`` is ``(n_frames, n_atoms, 3)``; ``box_list`` is
    ``(n_frames, 3, 3)`` or a single ``(3, 3)`` box (rows = box vectors).
    ``backend`` is ``"freud"`` or ``"rs"``. Frames with fewer pairs than the
    longest one are padded with ``pad_value`` (default ``n_atoms``, in all
    three columns, as in DMFF's per-frame padding). ``chunksize`` frames are
    sent to a worker per task; ``mp_context`` overrides the start method
    (default ``fork``); ``dtype`` defaults to int64, ``np.int32`` halves the
    memory of large batches.

    Returns an ``(n_frames, pmax, 3)`` array of ``[i, j, nbond]`` rows.
    """
    if backend not in _RAW_PAIR_BUILDERS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {list(_RAW_PAIR_BUILDERS)}")
    coords_list = np.asarray(coords_list)
    if coords_list.ndim != 3:
        raise ValueError("coords_list must have shape (n_frames, n_atoms, 3)")
    n_frames, n_atoms, _ = coords_list.shape
    box_list = np.asarray(box_list)
    if box_list.ndim == 2:
        box_list = np.broadcast_to(box_list, (n_frames, 3, 3))
    if box_list.shape != (n_frames, 3, 3):
        raise ValueError("box_list must have shape (n_frames, 3, 3) or (3, 3)")
    cov_map = np.asarray(cov_map)
    if pad_value is None:
        pad_value = n_atoms
    ncpu = os.cpu_count() or 1
    n_workers = ncpu if n_workers is None else int(n_workers)
    n_workers = max(1, min(n_workers, n_frames))

    func = functools.partial(_RAW_PAIR_BUILDERS[backend], rcut=float(rcut))
    if n_workers == 1:
        results = map(func, coords_list, box_list)
        pool = contextlib.nullcontext()
    else:
        # ``fork`` is the cheapest start method and, unlike spawn/forkserver,
        # does not re-run the user's (usually unguarded) script in the
        # children. The workers never touch jax, so JAX's "os.fork() ... is
        # multithreaded" RuntimeWarning does not apply and is silenced.
        if mp_context is None:
            mp_context = multiprocessing.get_context(
                "fork" if "fork" in multiprocessing.get_all_start_methods() else None
            )
        if chunksize is None:
            chunksize = max(1, n_frames // (n_workers * 4))
        pool = ProcessPoolExecutor(
            n_workers, mp_context=mp_context,
            initializer=_set_freud_threads, initargs=(max(1, ncpu // n_workers),),
        )
    with pool, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*os\.fork\(\) was called.*")
        if n_workers > 1:
            results = pool.map(func, coords_list, box_list, chunksize=chunksize)
        if show_progress:
            from tqdm import tqdm
            results = tqdm(results, total=n_frames, desc="Pair list")
        raw_pairs = list(results)

    # cov_map lookup + padding in the parent; only the tail of each frame is
    # padded (no full-array fill: the batch can be GBs).
    pmax = max((p.shape[0] for p in raw_pairs), default=0)
    pairs = np.empty((n_frames, pmax, 3), dtype=dtype)
    for i, p in enumerate(raw_pairs):
        m = p.shape[0]
        pairs[i, :m, :2] = p
        pairs[i, :m, 2] = cov_map[p[:, 0], p[:, 1]]
        pairs[i, m:, :] = pad_value
    return pairs


class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True):
        if freud is None:
            raise ImportError("Freud not installed.")
        self.box = np.asarray(box)
        self.fbox = freud.box.Box.from_matrix(self.box)
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
        nlist = _freud_raw_pairs(coords, box if box is not None else self.box, self.rcut)
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

    def allocate(self, coords, box=None):
        self._positions = coords

        cell_nm = self.box if box is None else np.asarray(box, dtype=np.float64)
        nlist = _rs_raw_pairs(coords, cell_nm, self.rcut)
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