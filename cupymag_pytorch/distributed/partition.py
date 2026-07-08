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
    # Misc
    # ------------------------------------------------------------------
    @property
    def owns_dof0(self):
        return self.r0 == 0

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
