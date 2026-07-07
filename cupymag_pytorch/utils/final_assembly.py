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

import numpy as np
import torch
from scipy.sparse import coo_matrix

from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.sparse_wrapper import (
    SparseMat,
    enforce_defect_region_A_scipy,
    enforce_defect_region_F_scipy,
)


def extract_defect_dofs(elements, global_id):
    """
    Identify the global DOFs of all nodes that belong to defect elements.

    Parameters
    ----------
    elements : torch.Tensor
        Element connectivity where the last column holds the defect flag.
    global_id : torch.Tensor
        Global DOF index for every node.

    Returns
    -------
    torch.Tensor
        Sorted unique global DOF indices of nodes inside defect elements.
    """
    defect_flags = elements[:, -1]
    mask_defect = defect_flags == 1
    if not bool(torch.any(mask_defect).item()):
        return torch.empty((0,), dtype=torch.int64, device=elements.device)

    n_nodes_per_elem = elements.shape[1] - 1
    defect_elems = elements[mask_defect, 0:n_nodes_per_elem]
    defect_nodes = torch.unique(defect_elems.reshape(-1))
    defect_dofs = torch.unique(global_id[defect_nodes])
    return defect_dofs


def assemble_stiffness_matrix(
    rows_np, cols_np, vals_np, nDOFx, defect_dofs=None, nDOFy=None
):
    """
    Assemble a CSR stiffness matrix on the device, with optional defect pinning.

    Parameters
    ----------
      rows_np, cols_np, vals_np: COO-format numpy arrays.
      defect_dofs: array of DOFs to pin to zero (defect region).
      nDOFx, nDOFy: matrix shape (nDOFy defaults to nDOFx).

    Returns
    -------
      SparseMat
    """
    if nDOFy is None:
        nDOFy = nDOFx

    if defect_dofs is not None and isinstance(defect_dofs, torch.Tensor):
        defect_dofs = defect_dofs.detach().cpu().numpy().astype(np.int64)

    A = coo_matrix(
        (
            np.asarray(vals_np, dtype=np.float64),
            (np.asarray(rows_np, dtype=np.int64), np.asarray(cols_np, dtype=np.int64)),
        ),
        shape=(nDOFx, nDOFy),
    ).tocsr()

    A = enforce_defect_region_A_scipy(A, defect_dofs)
    return SparseMat.from_scipy(A)


def assemble_mass_matrix(
    rows_np, cols_np, vals_np, nDOFx, defect_dofs=None, nDOFy=None
):
    """
    Assemble a CSR mass / derivative matrix on the device, with optional
    defect-region row zeroing.

    Returns
    -------
      SparseMat
    """
    if nDOFy is None:
        nDOFy = nDOFx

    if defect_dofs is not None and isinstance(defect_dofs, torch.Tensor):
        defect_dofs = defect_dofs.detach().cpu().numpy().astype(np.int64)

    F = coo_matrix(
        (
            np.asarray(vals_np, dtype=np.float64),
            (np.asarray(rows_np, dtype=np.int64), np.asarray(cols_np, dtype=np.int64)),
        ),
        shape=(nDOFx, nDOFy),
    ).tocsr()

    F = enforce_defect_region_F_scipy(F, defect_dofs)
    return SparseMat.from_scipy(F)


def build_E0_from_m(lam100, lam111, m_nodes):
    """
    Given ``m_nodes`` of shape (N, 3) (the magnetization at each node),
    return E0 of shape (6N,) containing the Voigt components
    (exx, eyy, ezz, 2*exy, 2*eyz, 2*exz) per node:

        E0_{ii} = (3/2) * lambda100 * (m_i^2 - 1/3)
        E0_{ij} = (3/2) * lambda111 * m_i m_j   (i != j)

    Stored in Voigt order per node:
        E0[6*n+0] = exx
        E0[6*n+1] = eyy
        E0[6*n+2] = ezz
        E0[6*n+3] = 2*exy
        E0[6*n+4] = 2*eyz
        E0[6*n+5] = 2*exz
    """
    if m_nodes.ndim == 1:
        N = m_nodes.size // 3
        m_nodes = m_nodes.reshape((N, 3))

    N = m_nodes.shape[0]
    E0 = torch.zeros((6 * N,), dtype=m_nodes.dtype, device=m_nodes.device)

    mx = m_nodes[:, 0]
    my = m_nodes[:, 1]
    mz = m_nodes[:, 2]

    one_third = 1.0 / 3.0
    three_over_2 = 1.5

    exx = three_over_2 * lam100 * (mx * mx - one_third)
    eyy = three_over_2 * lam100 * (my * my - one_third)
    ezz = three_over_2 * lam100 * (mz * mz - one_third)

    exy = three_over_2 * lam111 * (mx * my)
    eyz = three_over_2 * lam111 * (my * mz)
    exz = three_over_2 * lam111 * (mx * mz)

    E0[0::6] = exx
    E0[1::6] = eyy
    E0[2::6] = ezz
    E0[3::6] = 2.0 * exy
    E0[4::6] = 2.0 * eyz
    E0[5::6] = 2.0 * exz

    return E0
