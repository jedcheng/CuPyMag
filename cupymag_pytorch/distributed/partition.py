# Copyright (c) 2025-2026 Hongyi Guan
# PyTorch port of CuPyMag — distributed backend (Phase 1)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""x-slab partition of the periodic Hex FEM DOFs.

The compressed (periodic) DOF numbering produced by
``HexGrid.build_periodic_node_map`` is lexicographic with the x-plane
index slowest::

    dof(i, j, k) = i * Ny * Nz + j * Nz + k,   i in [0, Nx)

so a contiguous range of x-planes is a contiguous range of DOFs (and,
because ``gridHex`` emits elements x-column-major, a contiguous range of
elements). Trilinear hex elements couple only neighbouring planes, and
the mesh is periodic in x, so every rank needs exactly one ghost plane
from each ring neighbour.

Ghost-extended vector layout used by all distributed operators::

    [ owned DOFs (n_own) | left ghost plane (Ny*Nz) | right ghost plane (Ny*Nz) ]

The halo exchange is performed in two batched phases (rightward flow,
then leftward flow) so that each directed rank pair carries exactly one
message per phase — unambiguous even for world_size == 2, where both
neighbours are the same rank.
"""

import numpy as np
import torch
import torch.distributed as dist


class XSlabPartition:
    """Partition of ``Nx`` periodic DOF planes into contiguous rank slabs.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Element grid dimensions (== DOF plane count / plane shape after
        periodic compression).
    global_id_np : (n_nodes,) int array
        Node -> DOF map from ``build_periodic_node_map``; verified against
        the lexicographic layout this class relies on.
    device : torch.device
        Device for exchanged tensors.
    dtype : torch.dtype
        Dtype of matrix values / exchanged fields.
    """

    def __init__(self, Nx, Ny, Nz, global_id_np, device, dtype=torch.float64):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = device
        self.dtype = dtype

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.plane = Ny * Nz  # DOFs per x-plane == elements per x-column
        self.n_dof = Nx * self.plane

        self._verify_dof_layout(global_id_np)

        # Contiguous plane partition (same remainder rule as magnum.np).
        base, rem = divmod(Nx, self.world_size)
        counts = [base + (1 if r < rem else 0) for r in range(self.world_size)]
        starts = np.concatenate([[0], np.cumsum(counts)])
        if self.world_size > 1 and min(counts) < 2:
            raise RuntimeError(
                f"Each rank needs >= 2 x-planes (Nx={Nx}, ranks={self.world_size}); "
                "reduce the number of ranks."
            )
        self.plane_starts = starts  # length world_size + 1
        self.p0 = int(starts[self.rank])
        self.p1 = int(starts[self.rank + 1])

        # Owned DOF rows / element rows.
        self.r0 = self.p0 * self.plane
        self.r1 = self.p1 * self.plane
        self.n_own = self.r1 - self.r0
        self.e0 = self.p0 * self.plane
        self.e1 = self.p1 * self.plane

        # Ring neighbours (mesh is periodic in x).
        self.left = (self.rank - 1) % self.world_size
        self.right = (self.rank + 1) % self.world_size

        # Ghost plane global DOF ranges.
        lp = (self.p0 - 1) % Nx
        rp = self.p1 % Nx
        self.lg0 = lp * self.plane
        self.rg0 = rp * self.plane

        # Global DOF -> ghost-extended local index (-1 = unreachable).
        if self.world_size > 1:
            remap = np.full(self.n_dof, -1, dtype=np.int64)
            remap[self.r0 : self.r1] = np.arange(self.n_own)
            remap[self.lg0 : self.lg0 + self.plane] = self.n_own + np.arange(self.plane)
            remap[self.rg0 : self.rg0 + self.plane] = (
                self.n_own + self.plane + np.arange(self.plane)
            )
            self.n_ext = self.n_own + 2 * self.plane
        else:
            remap = np.arange(self.n_dof, dtype=np.int64)
            self.n_ext = self.n_dof
        self._remap = remap

        # Global DOF -> (own + right ghost) index, for element-based
        # operations (volume averages) that only need the plane above.
        if self.world_size > 1:
            remap_r = np.full(self.n_dof, -1, dtype=np.int64)
            remap_r[self.r0 : self.r1] = np.arange(self.n_own)
            remap_r[self.rg0 : self.rg0 + self.plane] = self.n_own + np.arange(
                self.plane
            )
        else:
            remap_r = remap
        self._remap_right = remap_r

    # ------------------------------------------------------------------
    # Layout verification
    # ------------------------------------------------------------------
    def _verify_dof_layout(self, global_id_np):
        """Assert dof(i,j,k) == (i%Nx)*Ny*Nz + (j%Ny)*Nz + (k%Nz)."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        i, j, k = np.meshgrid(
            np.arange(Nx + 1), np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij"
        )
        expected = ((i % Nx) * Ny * Nz + (j % Ny) * Nz + (k % Nz)).ravel()
        gid = np.asarray(global_id_np).ravel()
        if gid.shape != expected.shape or not np.array_equal(gid, expected):
            raise RuntimeError(
                "Periodic DOF numbering does not match the lexicographic "
                "x-slab layout required by the distributed backend."
            )

    # ------------------------------------------------------------------
    # Row-block extraction
    # ------------------------------------------------------------------
    def _to_torch_csr(self, B):
        crow = torch.from_numpy(np.ascontiguousarray(B.indptr, dtype=np.int64)).to(
            self.device
        )
        col = torch.from_numpy(np.ascontiguousarray(B.indices, dtype=np.int64)).to(
            self.device
        )
        vals = (
            torch.from_numpy(np.ascontiguousarray(B.data, dtype=np.float64))
            .to(self.dtype)
            .to(self.device)
        )
        return torch.sparse_csr_tensor(
            crow, col, vals, B.shape, requires_grad=False
        )

    def remap_strided(self, stride):
        """Column remap for operators whose columns are node-major with
        ``stride`` entries per node (e.g. F_el with 6 Voigt components):
        global column node*stride + j -> local ext-node index * stride + j."""
        if stride == 1:
            return self._remap
        rep = np.repeat(self._remap, stride)
        offs = np.tile(np.arange(stride, dtype=np.int64), self.n_dof)
        return np.where(rep >= 0, rep * stride + offs, -1)

    def extract_rows(self, A_scipy_csr, row_start=0, col_stride=1):
        """Extract the rows [row_start + r0, row_start + r1) of a global
        scipy CSR matrix whose columns are node-major with ``col_stride``
        entries per node, remapping columns into the ghost-extended local
        index space and splitting by column ownership.

        Returns ``(A_own, A_ghost)`` torch sparse CSR tensors of shapes
        (n_own, n_own*col_stride) and (n_own, 2*plane*col_stride);
        ``A_ghost`` is ``None`` for a single rank. The split lets the ghost
        exchange overlap with the ``A_own`` SpMV.
        """
        from scipy.sparse import csr_matrix

        B = A_scipy_csr[row_start + self.r0 : row_start + self.r1, :].tocsr()
        remap = self.remap_strided(col_stride)
        cols = remap[B.indices]
        if cols.size and cols.min() < 0:
            bad = np.unique(B.indices[cols < 0])
            raise RuntimeError(
                f"Rank {self.rank}: matrix couples owned rows to DOFs outside "
                f"the +-1-plane halo (e.g. global cols {bad[:10]}); x-slab "
                "partitioning is not valid for this operator."
            )
        n_own_c = self.n_own * col_stride
        n_ext_c = self.n_ext * col_stride
        Bm = csr_matrix((B.data, cols, B.indptr), shape=(self.n_own, n_ext_c))
        if self.world_size == 1:
            return self._to_torch_csr(Bm), None
        A_own = self._to_torch_csr(Bm[:, :n_own_c].tocsr())
        A_ghost = self._to_torch_csr(Bm[:, n_own_c:].tocsr())
        return A_own, A_ghost

    def extract_row_block(self, A_scipy_csr):
        """Row block [r0, r1) of a global (n_dof x n_dof) scipy CSR matrix
        (scalar node columns); see :meth:`extract_rows`."""
        return self.extract_rows(A_scipy_csr)

    # ------------------------------------------------------------------
    # Halo exchange
    # ------------------------------------------------------------------
    def _p2p_round(self, ops):
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    def ghosts_start(self, x):
        """Begin the halo exchange for 2-D ``x`` (n_own, k); returns an
        opaque context for :meth:`ghosts_finish`.

        For world_size >= 3 each directed rank pair carries one message,
        so all four P2P ops are posted in a single non-blocking batch that
        can overlap with local computation. For world_size == 2 both
        neighbours are the same rank and the messages would cross-match,
        so the two flow directions run as sequential blocking phases.
        """
        ps = self.plane
        if self.world_size == 1:
            return None

        gl = torch.empty((ps, x.shape[1]), dtype=x.dtype, device=x.device)
        gr = torch.empty_like(gl)
        first = x[:ps].contiguous()
        last = x[-ps:].contiguous()

        if self.world_size == 2:
            # Phase A: rightward flow; Phase B: leftward flow (blocking).
            self._p2p_round(
                [
                    dist.P2POp(dist.isend, last, self.right),
                    dist.P2POp(dist.irecv, gl, self.left),
                ]
            )
            self._p2p_round(
                [
                    dist.P2POp(dist.isend, first, self.left),
                    dist.P2POp(dist.irecv, gr, self.right),
                ]
            )
            return (None, gl, gr, first, last)

        reqs = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, last, self.right),
                dist.P2POp(dist.irecv, gl, self.left),
                dist.P2POp(dist.isend, first, self.left),
                dist.P2POp(dist.irecv, gr, self.right),
            ]
        )
        # Keep references to the send buffers until the wait.
        return (reqs, gl, gr, first, last)

    def ghosts_finish(self, ctx):
        """Complete a :meth:`ghosts_start` exchange; returns the (2*plane, k)
        ghost block [left plane; right plane], or ``None`` for 1 rank."""
        if ctx is None:
            return None
        reqs, gl, gr, _, _ = ctx
        if reqs is not None:
            for req in reqs:
                req.wait()
        return torch.cat([gl, gr], dim=0)

    def get_ghosts(self, x):
        """Blocking halo exchange for 2-D ``x``: returns the (2*plane, k)
        ghost block, or ``None`` for 1 rank."""
        return self.ghosts_finish(self.ghosts_start(x))

    def exchange_ghosts(self, x):
        """Return the ghost-extended vector [x | left ghost | right ghost].

        ``x`` is (n_own,) or (n_own, k). The left ghost is the left
        neighbour's last owned plane; the right ghost is the right
        neighbour's first owned plane (ring, periodic in x).
        """
        if self.world_size == 1:
            return x

        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(1)

        ghosts = self.get_ghosts(x)
        x_ext = torch.cat([x, ghosts], dim=0)
        return x_ext.squeeze(1) if squeeze else x_ext

    def extend_right(self, x):
        """Return [x | right ghost plane] for element-based operations."""
        if self.world_size == 1:
            return x

        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(1)

        ps = self.plane
        gr = torch.empty((ps, x.shape[1]), dtype=x.dtype, device=x.device)
        self._p2p_round(
            [
                dist.P2POp(dist.isend, x[:ps].contiguous(), self.left),
                dist.P2POp(dist.irecv, gr, self.right),
            ]
        )
        x_ext = torch.cat([x, gr], dim=0)
        return x_ext.squeeze(1) if squeeze else x_ext

    # ------------------------------------------------------------------
    # Gather (for rank-0 I/O)
    # ------------------------------------------------------------------
    def gather_rows(self, x_local):
        """Gather row-distributed data to the full (n_dof, ...) tensor.

        Returns the global tensor on rank 0 and ``None`` elsewhere.
        """
        if self.world_size == 1:
            return x_local.clone()

        squeeze = x_local.dim() == 1
        if squeeze:
            x_local = x_local.unsqueeze(1)

        counts = [
            int(self.plane_starts[r + 1] - self.plane_starts[r]) * self.plane
            for r in range(self.world_size)
        ]
        max_n = max(counts)
        padded = torch.zeros(
            (max_n, x_local.shape[1]), dtype=x_local.dtype, device=x_local.device
        )
        padded[: x_local.shape[0]] = x_local

        chunks = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(chunks, padded)

        if self.rank != 0:
            return None

        full = torch.empty(
            (self.n_dof, x_local.shape[1]), dtype=x_local.dtype, device=x_local.device
        )
        off = 0
        for r in range(self.world_size):
            full[off : off + counts[r]] = chunks[r][: counts[r]]
            off += counts[r]
        return full.squeeze(1) if squeeze else full

    # ------------------------------------------------------------------
    # Misc / generic partition interface
    # ------------------------------------------------------------------
    @property
    def owns_dof0(self):
        return self.r0 == 0

    # Anchor DOF (global DOF 0) location — plane 0 lives on rank 0.
    anchor_owner = 0
    anchor_local = 0

    def slice_field(self, full):
        """This rank's rows of a full (n_dof, ...) field in DOF order."""
        return full[self.r0 : self.r1].clone()

    def slice_rows_np(self, vec):
        """This rank's rows of a full (n_dof,) numpy vector in DOF order."""
        return np.asarray(vec)[self.r0 : self.r1]

    @property
    def element_remap(self):
        """Global DOF -> local index map covering the DOFs referenced by
        this rank's elements (owned + right ghost plane)."""
        return self._remap_right

    def extend_elements(self, x):
        """Ghost-extend a local field to cover this rank's elements."""
        return self.extend_right(x)

    def select_elements(self, elements):
        """This rank's element rows."""
        return elements[self.e0 : self.e1]

    def local_defect_dofs(self, defect_dofs_global):
        """Slice global defect DOF indices to owned local indices."""
        d = defect_dofs_global
        if isinstance(d, torch.Tensor):
            mask = (d >= self.r0) & (d < self.r1)
            return (d[mask] - self.r0).to(torch.int64)
        d = np.asarray(d)
        return torch.as_tensor(
            d[(d >= self.r0) & (d < self.r1)] - self.r0,
            dtype=torch.int64,
            device=self.device,
        )


class GeneralPartition:
    """Row partition for unstructured (Tet) meshes.

    DOFs are renumbered by a lexicographic coordinate sort (x slowest, the
    unstructured analogue of the Hex x-slab layout) so that each rank owns
    a contiguous range of the *permuted* numbering. Ghost DOFs are the
    element-adjacency neighbours of the owned DOFs — exactly the coupling
    stencil of every FEM operator — and are exchanged via per-neighbour
    index lists. Each directed rank pair carries exactly one message per
    exchange, so a single non-blocking batch is unambiguous at any world
    size. A graph partitioner (e.g. METIS) can replace the coordinate sort
    by supplying a different permutation; everything downstream only
    assumes contiguous ranges in the permuted space.

    Exposes the same interface as :class:`XSlabPartition` (extract_rows,
    ghosts_start/finish/get_ghosts, exchange_ghosts, gather_rows,
    slice_field, element helpers, anchor location).
    """

    def __init__(self, mesh, device, dtype=torch.float64):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = device
        self.dtype = dtype

        gid = np.asarray(mesh.global_id_np, dtype=np.int64)
        coords = np.asarray(mesh.node_coords_np, dtype=np.float64)
        elements = np.asarray(mesh.elements_np)
        n_per_elem = elements.shape[1] - 1
        self.n_dof = int(gid.max()) + 1

        # One representative node per DOF (first encounter, matching the
        # periodic-map construction), for the coordinate sort.
        first_node = np.full(self.n_dof, len(gid), dtype=np.int64)
        np.minimum.at(first_node, gid, np.arange(len(gid), dtype=np.int64))
        dc = coords[first_node]

        # perm[new_id] = old DOF id; iperm[old DOF id] = new_id.
        self.perm = np.lexsort((dc[:, 2], dc[:, 1], dc[:, 0])).astype(np.int64)
        self.iperm = np.empty_like(self.perm)
        self.iperm[self.perm] = np.arange(self.n_dof, dtype=np.int64)

        base, rem = divmod(self.n_dof, self.world_size)
        counts = [base + (1 if r < rem else 0) for r in range(self.world_size)]
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.range_starts = starts
        self.r0 = int(starts[self.rank])
        self.r1 = int(starts[self.rank + 1])
        self.n_own = self.r1 - self.r0

        # Ghosts per rank from element adjacency (replicated computation).
        corner_new = self.iperm[gid[elements[:, :n_per_elem].astype(np.int64)]]
        owner = np.searchsorted(starts, corner_new, side="right") - 1
        ghost_lists = []
        for r in range(self.world_size):
            touch = (owner == r).any(axis=1)
            cs = np.unique(corner_new[touch])
            ghost_lists.append(cs[(cs < starts[r]) | (cs >= starts[r + 1])])
        self.ghosts = ghost_lists[self.rank]  # sorted permuted ids
        self.n_ghost = len(self.ghosts)
        self.n_ext = self.n_own + self.n_ghost

        # Receive segments (my sorted ghost list is grouped by owner) and
        # send index lists (what each other rank needs from my range).
        self._recv_seg = {}
        for o in range(self.world_size):
            if o == self.rank:
                continue
            lo, hi = np.searchsorted(self.ghosts, [starts[o], starts[o + 1]])
            if hi > lo:
                self._recv_seg[o] = (int(lo), int(hi))
        self._send_idx = {}
        for o in range(self.world_size):
            if o == self.rank:
                continue
            need = ghost_lists[o]
            mine = need[(need >= self.r0) & (need < self.r1)] - self.r0
            if mine.size:
                self._send_idx[o] = torch.as_tensor(mine, device=device)
        self._peers = sorted(set(self._recv_seg) | set(self._send_idx))

        # Global DOF (original numbering) -> ghost-extended local index.
        remap = np.full(self.n_dof, -1, dtype=np.int64)
        remap[self.perm[self.r0 : self.r1]] = np.arange(self.n_own)
        if self.n_ghost:
            remap[self.perm[self.ghosts]] = self.n_own + np.arange(self.n_ghost)
        self._remap_g = remap

        # Element ownership: rank owning the minimum permuted corner id
        # (guarantees all corners are in own + ghost).
        min_new = corner_new.min(axis=1)
        eown = np.searchsorted(starts, min_new, side="right") - 1
        self._elem_idx = torch.as_tensor(
            np.where(eown == self.rank)[0], dtype=torch.int64, device=device
        )

        a_new = int(self.iperm[0])
        self.anchor_owner = int(np.searchsorted(starts, a_new, side="right") - 1)
        self.anchor_local = a_new - int(starts[self.anchor_owner])

        self._perm_t = torch.as_tensor(self.perm, device=device)
        self._iperm_t = torch.as_tensor(self.iperm, device=device)

        if self.rank == 0:
            counts_g = [len(g) for g in ghost_lists]
            print(
                f"[GeneralPartition] {self.n_dof} DOFs over "
                f"{self.world_size} ranks (~{counts[0]} each), ghost counts "
                f"{counts_g}."
            )

    # ------------------------------------------------------------------
    # Row-block extraction (columns remapped by original DOF id)
    # ------------------------------------------------------------------
    _to_torch_csr = XSlabPartition._to_torch_csr

    def remap_strided(self, stride):
        if stride == 1:
            return self._remap_g
        rep = np.repeat(self._remap_g, stride)
        offs = np.tile(np.arange(stride, dtype=np.int64), self.n_dof)
        return np.where(rep >= 0, rep * stride + offs, -1)

    def extract_rows(self, A_scipy_csr, row_start=0, col_stride=1):
        from scipy.sparse import csr_matrix

        rows = row_start + self.perm[self.r0 : self.r1]
        B = A_scipy_csr[rows, :].tocsr()
        cols = self.remap_strided(col_stride)[B.indices]
        if cols.size and cols.min() < 0:
            bad = np.unique(B.indices[cols < 0])
            raise RuntimeError(
                f"Rank {self.rank}: matrix couples owned rows to DOFs outside "
                f"the element-adjacency ghost set (e.g. global cols {bad[:10]})."
            )
        n_own_c = self.n_own * col_stride
        n_ext_c = self.n_ext * col_stride
        Bm = csr_matrix((B.data, cols, B.indptr), shape=(self.n_own, n_ext_c))
        if self.world_size == 1:
            return self._to_torch_csr(Bm), None
        A_own = self._to_torch_csr(Bm[:, :n_own_c].tocsr())
        A_ghost = self._to_torch_csr(Bm[:, n_own_c:].tocsr())
        return A_own, A_ghost

    def extract_row_block(self, A_scipy_csr):
        return self.extract_rows(A_scipy_csr)

    # ------------------------------------------------------------------
    # Halo exchange (index lists, one message per directed pair)
    # ------------------------------------------------------------------
    def ghosts_start(self, x):
        if self.world_size == 1:
            return None
        if not self._peers:
            return ("noop", None, None, x.shape[1], x.dtype, x.device)
        bufs = {}
        sends = {}
        ops = []
        for o in self._peers:
            if o in self._send_idx:
                sends[o] = x[self._send_idx[o]].contiguous()
                ops.append(dist.P2POp(dist.isend, sends[o], o))
            if o in self._recv_seg:
                lo, hi = self._recv_seg[o]
                bufs[o] = torch.empty((hi - lo, x.shape[1]), dtype=x.dtype, device=x.device)
                ops.append(dist.P2POp(dist.irecv, bufs[o], o))
        reqs = dist.batch_isend_irecv(ops)
        return (reqs, bufs, sends, x.shape[1], x.dtype, x.device)

    def ghosts_finish(self, ctx):
        if ctx is None:
            return None
        reqs, bufs, _, k, dtype, device = ctx
        if reqs == "noop":
            return torch.empty((self.n_ghost, k), dtype=dtype, device=device)
        for req in reqs:
            req.wait()
        ghosts = torch.empty((self.n_ghost, k), dtype=dtype, device=device)
        for o, (lo, hi) in self._recv_seg.items():
            ghosts[lo:hi] = bufs[o]
        return ghosts

    def get_ghosts(self, x):
        return self.ghosts_finish(self.ghosts_start(x))

    def exchange_ghosts(self, x):
        if self.world_size == 1:
            return x
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(1)
        ghosts = self.get_ghosts(x)
        x_ext = torch.cat([x, ghosts], dim=0) if ghosts is not None else x
        return x_ext.squeeze(1) if squeeze else x_ext

    # ------------------------------------------------------------------
    # Generic partition interface
    # ------------------------------------------------------------------
    def slice_field(self, full):
        return full[self._perm_t[self.r0 : self.r1]].clone()

    def slice_rows_np(self, vec):
        return np.asarray(vec)[self.perm[self.r0 : self.r1]]

    @property
    def element_remap(self):
        return self._remap_g

    def extend_elements(self, x):
        return self.exchange_ghosts(x)

    def select_elements(self, elements):
        return elements[self._elem_idx]

    def local_defect_dofs(self, defect_dofs_global):
        d = defect_dofs_global
        if isinstance(d, torch.Tensor):
            new = self._iperm_t[d.to(torch.int64)]
        else:
            new = torch.as_tensor(
                self.iperm[np.asarray(d)], dtype=torch.int64, device=self.device
            )
        mask = (new >= self.r0) & (new < self.r1)
        return (new[mask] - self.r0).to(torch.int64)

    def gather_rows(self, x_local):
        """Gather row-distributed data into the full tensor in the
        *original* DOF order. Returns on rank 0, ``None`` elsewhere."""
        if self.world_size == 1:
            return x_local.clone()

        squeeze = x_local.dim() == 1
        if squeeze:
            x_local = x_local.unsqueeze(1)

        counts = [
            int(self.range_starts[r + 1] - self.range_starts[r])
            for r in range(self.world_size)
        ]
        max_n = max(counts)
        padded = torch.zeros(
            (max_n, x_local.shape[1]), dtype=x_local.dtype, device=x_local.device
        )
        padded[: x_local.shape[0]] = x_local

        chunks = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(chunks, padded)

        if self.rank != 0:
            return None

        full_perm = torch.empty(
            (self.n_dof, x_local.shape[1]), dtype=x_local.dtype, device=x_local.device
        )
        off = 0
        for r in range(self.world_size):
            full_perm[off : off + counts[r]] = chunks[r][: counts[r]]
            off += counts[r]
        full = torch.empty_like(full_perm)
        full[self._perm_t] = full_perm  # back to original DOF order
        return full.squeeze(1) if squeeze else full
