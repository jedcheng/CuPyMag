# Copyright (c) 2025-2026 Hongyi Guan
# PyTorch port of CuPyMag
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

"""Sparse matrix utilities for the PyTorch port of CuPyMag.

The original code used ``cupyx.scipy.sparse`` CSR matrices with the ``@``
operator for sparse matrix-vector products. PyTorch provides
``torch.sparse_csr_tensor`` plus ``torch.sparse.mm`` for the same
operation; this module wraps them in a tiny ``SparseMat`` class that
keeps the ``A @ b`` calling convention identical.

Duplicate (row, col) entries are summed during construction (mirroring
``scipy.sparse.coo_matrix`` semantics), and defect/anchor DOF pinning is
performed once, on the CPU, before the matrix is moved to the device.
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix

from cupymag_pytorch.utils.backend import DEVICE


class SparseMat:
    """A sparse CSR matrix living on ``DEVICE``.

    Supports the ``A @ b`` operation where ``b`` is a 1-D or 2-D dense
    torch tensor. Internally backed by a ``torch.sparse_csr_tensor``.
    """

    def __init__(self, csr_tensor: torch.Tensor):
        if not (csr_tensor.is_sparse or csr_tensor.is_sparse_csr):
            raise TypeError("SparseMat expects a sparse (torch.sparse_csr) tensor.")
        self.t = csr_tensor

    # -- construction -----------------------------------------------------
    @classmethod
    def from_scipy(
        cls, A: csr_matrix, dtype=torch.float64, device=DEVICE
    ) -> "SparseMat":
        """Build a SparseMat from a scipy CSR matrix."""
        crow = torch.from_numpy(np.ascontiguousarray(A.indptr, dtype=np.int64)).to(
            device
        )
        col = torch.from_numpy(np.ascontiguousarray(A.indices, dtype=np.int64)).to(
            device
        )
        values = (
            torch.from_numpy(np.ascontiguousarray(A.data, dtype=np.float64))
            .to(dtype)
            .to(device)
        )
        # scipy CSR has already summed duplicate (row, col) entries, so the
        # resulting torch CSR tensor is canonical and needs no coalescing.
        t = torch.sparse_csr_tensor(crow, col, values, A.shape, requires_grad=False)
        return cls(t)

    @classmethod
    def from_coo(
        cls, rows, cols, vals, shape, dtype=torch.float64, device=DEVICE
    ) -> "SparseMat":
        """Build a SparseMat from COO-format arrays (duplicates summed)."""
        A = coo_matrix(
            (
                np.asarray(vals, dtype=np.float64),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=shape,
        ).tocsr()
        return cls.from_scipy(A, dtype=dtype, device=device)

    # -- properties / ops -------------------------------------------------
    @property
    def shape(self):
        return self.t.shape

    def diagonal(self):
        """Return the main diagonal as a dense 1-D tensor."""
        coo = self.t.to_sparse_coo().coalesce()
        idx = coo.indices()
        mask = idx[0] == idx[1]
        d = torch.zeros(
            min(self.shape), dtype=coo.values().dtype, device=coo.values().device
        )
        d[idx[0][mask]] = coo.values()[mask]
        return d

    def matmul(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, device=self.t.device, dtype=self.t.dtype)
        v = v.to(self.t.device)
        if v.dim() == 1:
            return torch.sparse.mm(self.t, v.unsqueeze(1)).squeeze(1)
        return torch.sparse.mm(self.t, v)

    def __matmul__(self, v):
        return self.matmul(v)


# ----------------------------------------------------------------------
# Defect-region enforcement (operates on scipy CSR, runs once at assembly)
# ----------------------------------------------------------------------
def enforce_defect_region_A_scipy(A: csr_matrix, defect_dofs) -> csr_matrix:
    """Pin defect DOFs in the stiffness matrix.

    Zeros out every entry whose row or column touches a defect DOF, then
    inserts unit diagonal entries so that x[defect] = b[defect] = 0.
    """
    if defect_dofs is None:
        return A
    ddofs = np.asarray(defect_dofs, dtype=np.int64)
    if ddofs.size == 0:
        return A

    A = A.tocoo()
    mask = ~(np.isin(A.row, ddofs) | np.isin(A.col, ddofs))
    rows = np.concatenate([A.row[mask], ddofs])
    cols = np.concatenate([A.col[mask], ddofs])
    vals = np.concatenate([A.data[mask], np.ones(ddofs.shape[0], dtype=np.float64)])
    return coo_matrix((vals, (rows, cols)), shape=A.shape).tocsr()


def enforce_defect_region_F_scipy(F: csr_matrix, defect_dofs) -> csr_matrix:
    """Zero out the rows of the mass/derivative matrix that touch defect DOFs.

    This makes the corresponding right-hand-side entries ``F @ m`` zero,
    consistent with m = 0 inside the defect region.
    """
    if defect_dofs is None:
        return F
    ddofs = np.asarray(defect_dofs, dtype=np.int64)
    if ddofs.size == 0:
        return F

    F = F.tocoo()
    mask = ~np.isin(F.row, ddofs)
    return coo_matrix((F.data[mask], (F.row[mask], F.col[mask])), shape=F.shape).tocsr()
