# Copyright (c) 2025-2026 Hongyi Guan
# This file is part of CuPyMag
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

from cupymag_pytorch.core.constants import *
from cupymag_pytorch.core.parameters import grid_type, rot111
from cupymag_pytorch.utils.numba_shim import float64, int32, int64, jitclass, njit
from cupymag_pytorch.utils.rot_111_matrices import get_M_matrix

if grid_type == "Hex":
    from cupymag_pytorch.mesh.gridHex import HexGrid
    from cupymag_pytorch.mesh.ShapeHex import (
        element_jacobian,
        gauss_quadrature,
        get_dN,
        get_N,
    )
elif grid_type == "Tet":
    from cupymag_pytorch.mesh.gridTet import TetraGrid
    from cupymag_pytorch.mesh.ShapeTet import (
        element_jacobian,
        gauss_quadrature,
        get_dN,
        get_N,
    )
else:
    raise NotImplementedError(f"Grid type '{grid_type}' is not supported.")


int_numba = int32
int_np = np.int32

spec = {
    "node_coords": float64[:, :],
    "elements": int_numba[:, :],
    "Nnodes_per_element": int_numba,
    "gauss_points": float64[:, :],
    "gauss_weights": float64[:],
    "global_id": int_numba[:],
    "anchor_row": int64,
    "c11": float64,
    "c12": float64,
    "c44": float64,
    "lambda100": float64,
    "lambda111": float64,
    "nNodeGlobal": int_numba,
    "C_voigt": float64[:, :],
}

_M_mat = get_M_matrix(backend="numpy")


@njit
def _transform_voigt(C_voigt, M):
    """
    Transform a 6x6 Voigt elasticity matrix: C' = M C M^T.
    """
    tmp = np.zeros((6, 6), dtype=np.float64)
    out = np.zeros((6, 6), dtype=np.float64)

    # tmp = M @ C_voigt
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += M[i, k] * C_voigt[k, j]
            tmp[i, j] = s

    # out = tmp @ M.T
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += tmp[i, k] * M[j, k]
            out[i, j] = s

    return out


@jitclass(spec)
class AssembleElasticity:
    """
    Finite element assembly class for mechanical equilibrium with linear
    elasticity and magneto-elastic coupling.

    Uses the standard B^T C B stiffness assembly and supports an optional
    crystal orientation rotation from <100> to <111>.
    """

    def __init__(
        self, node_coords, elements, global_id, c11, c12, c44, lambda100, lambda111
    ):
        self.node_coords = node_coords
        self.elements = elements
        self.Nnodes_per_element = self.elements.shape[1] - 1
        self.gauss_points, self.gauss_weights = gauss_quadrature()
        self.global_id = global_id
        self.anchor_row = 0

        self.c11 = c11
        self.c12 = c12
        self.c44 = c44
        self.lambda100 = lambda100
        self.lambda111 = lambda111

        self.nNodeGlobal = self.global_id.max() + 1

        self.C_voigt = np.zeros((6, 6), dtype=np.float64)
        self.C_voigt[0, 0] = self.c11
        self.C_voigt[0, 1] = self.c12
        self.C_voigt[0, 2] = self.c12
        self.C_voigt[1, 0] = self.c12
        self.C_voigt[1, 1] = self.c11
        self.C_voigt[1, 2] = self.c12
        self.C_voigt[2, 0] = self.c12
        self.C_voigt[2, 1] = self.c12
        self.C_voigt[2, 2] = self.c11
        self.C_voigt[3, 3] = self.c44
        self.C_voigt[4, 4] = self.c44
        self.C_voigt[5, 5] = self.c44

        if rot111 == True:
            M = _M_mat
            self.C_voigt = _transform_voigt(self.C_voigt, M)

    def compute_element_stiffness_elasticity(self, xc, yc, zc):
        """Element stiffness matrix K_e via B^T C B (3 DOFs per node)."""
        nNodes = self.Nnodes_per_element
        ngp = len(self.gauss_points)

        K_e = np.zeros((3 * nNodes, 3 * nNodes), dtype=np.float64)

        for igp in range(ngp):
            w = self.gauss_weights[igp]

            J_ = element_jacobian(xc, yc, zc, igp)

            detJ = np.linalg.det(J_)
            if abs(detJ) < 1e-14:
                return np.zeros((3 * nNodes, 3 * nNodes), dtype=np.float64)

            J_inv = np.linalg.inv(J_)

            dN_rst = get_dN(igp)

            B = np.zeros((6, 3 * nNodes), dtype=np.float64)

            for a in range(nNodes):
                dNx = (
                    J_inv[0, 0] * dN_rst[a, 0]
                    + J_inv[0, 1] * dN_rst[a, 1]
                    + J_inv[0, 2] * dN_rst[a, 2]
                )
                dNy = (
                    J_inv[1, 0] * dN_rst[a, 0]
                    + J_inv[1, 1] * dN_rst[a, 1]
                    + J_inv[1, 2] * dN_rst[a, 2]
                )
                dNz = (
                    J_inv[2, 0] * dN_rst[a, 0]
                    + J_inv[2, 1] * dN_rst[a, 1]
                    + J_inv[2, 2] * dN_rst[a, 2]
                )

                colA = 3 * a

                B[0, colA + 0] = dNx
                B[1, colA + 1] = dNy
                B[2, colA + 2] = dNz

                B[3, colA + 0] = dNy
                B[3, colA + 1] = dNx

                B[4, colA + 1] = dNz
                B[4, colA + 2] = dNy

                B[5, colA + 0] = dNz
                B[5, colA + 2] = dNx

            BT_C = np.zeros((3 * nNodes, 6), dtype=np.float64)
            for i in range(3 * nNodes):
                for j in range(6):
                    for k in range(6):
                        BT_C[i, j] += B[k, i] * self.C_voigt[k, j]

            for i in range(3 * nNodes):
                for j in range(3 * nNodes):
                    for k in range(6):
                        K_e[i, j] += BT_C[i, k] * B[k, j] * (detJ * w)

        return K_e

    def compute_element_F(self, xc, yc, zc):
        """Element coupling matrix Fe relating spontaneous strains to displacements."""
        nNodes = self.Nnodes_per_element
        ngp = len(self.gauss_points)

        Fe = np.zeros((3 * nNodes, 6 * nNodes), dtype=np.float64)

        for igp in range(ngp):
            J_ = element_jacobian(xc, yc, zc, igp)

            detJ = np.linalg.det(J_)
            if abs(detJ) < 1e-14:
                return np.zeros((3 * nNodes, 6 * nNodes), dtype=np.float64)

            J_inv = np.linalg.inv(J_)

            w = self.gauss_weights[igp]

            dN_rst = get_dN(igp)
            N_rst = get_N(igp)

            B = np.zeros((6, 3 * nNodes), dtype=np.float64)

            for a in range(nNodes):
                dNx = (
                    J_inv[0, 0] * dN_rst[a, 0]
                    + J_inv[0, 1] * dN_rst[a, 1]
                    + J_inv[0, 2] * dN_rst[a, 2]
                )
                dNy = (
                    J_inv[1, 0] * dN_rst[a, 0]
                    + J_inv[1, 1] * dN_rst[a, 1]
                    + J_inv[1, 2] * dN_rst[a, 2]
                )
                dNz = (
                    J_inv[2, 0] * dN_rst[a, 0]
                    + J_inv[2, 1] * dN_rst[a, 1]
                    + J_inv[2, 2] * dN_rst[a, 2]
                )

                colA = 3 * a

                B[0, colA + 0] = dNx
                B[1, colA + 1] = dNy
                B[2, colA + 2] = dNz

                B[3, colA + 0] = dNy
                B[3, colA + 1] = dNx

                B[4, colA + 1] = dNz
                B[4, colA + 2] = dNy

                B[5, colA + 0] = dNz
                B[5, colA + 2] = dNx

            S = np.zeros((6, 6 * nNodes), dtype=np.float64)

            for a in range(nNodes):
                N_a = N_rst[a]
                for v in range(6):
                    S[v, 6 * a + v] = N_a

            BT_C = np.zeros((3 * nNodes, 6), dtype=np.float64)
            for i in range(3 * nNodes):
                for j in range(6):
                    for k in range(6):
                        BT_C[i, j] += B[k, i] * self.C_voigt[k, j]

            for i in range(3 * nNodes):
                for j in range(6 * nNodes):
                    for k in range(6):
                        Fe[i, j] += BT_C[i, k] * S[k, j] * (detJ * w)

        return Fe

    def build_coo_matrix_A_numba(self):
        """Build the global elasticity stiffness matrix in COO format (3x3 block structure)."""
        node_coords_np = self.node_coords
        elements_np = self.elements
        global_id_np = self.global_id

        Ne = elements_np.shape[0]
        nNodes = self.Nnodes_per_element
        nnz = Ne * (3 * nNodes) ** 2

        rows_np = np.empty(nnz, dtype=int_np)
        cols_np = np.empty(nnz, dtype=int_np)
        vals_np = np.zeros(nnz, dtype=np.float64)

        idx = 0
        nNodeGlobal = self.nNodeGlobal

        for e in range(Ne):
            corner_ids = elements_np[e, 0:nNodes]

            xc = np.zeros(nNodes, dtype=np.float64)
            yc = np.zeros(nNodes, dtype=np.float64)
            zc = np.zeros(nNodes, dtype=np.float64)

            for i in range(nNodes):
                nid = corner_ids[i]
                xc[i] = node_coords_np[nid, 0]
                yc[i] = node_coords_np[nid, 1]
                zc[i] = node_coords_np[nid, 2]

            K_e = self.compute_element_stiffness_elasticity(xc, yc, zc)

            for a in range(nNodes):
                for da in range(3):
                    rowA = global_id_np[corner_ids[a]] + da * nNodeGlobal
                    for b in range(nNodes):
                        for db in range(3):
                            colB = global_id_np[corner_ids[b]] + db * nNodeGlobal
                            rows_np[idx] = rowA
                            cols_np[idx] = colB
                            vals_np[idx] = K_e[3 * a + da, 3 * b + db]
                            idx += 1

        rows_np, cols_np, vals_np = self.impose_anchor_node_dof0_coo(
            rows_np, cols_np, vals_np
        )

        return rows_np, cols_np, vals_np

    def build_coo_matrices_F_numba(self):
        """Build the global coupling matrix F in COO format."""
        node_coords_np = self.node_coords
        elements_np = self.elements
        global_id_np = self.global_id

        Ne = elements_np.shape[0]
        nNodes = self.Nnodes_per_element
        nnz = Ne * (3 * nNodes) * (6 * nNodes)

        rows_np = np.empty(nnz, dtype=int_np)
        cols_np = np.empty(nnz, dtype=int_np)
        vals_np = np.zeros(nnz, dtype=np.float64)

        idx = 0
        nNodeGlobal = self.nNodeGlobal

        for e in range(Ne):
            corner_ids = elements_np[e, 0:nNodes]

            xc = np.zeros(nNodes, dtype=np.float64)
            yc = np.zeros(nNodes, dtype=np.float64)
            zc = np.zeros(nNodes, dtype=np.float64)

            for i in range(nNodes):
                nid = corner_ids[i]
                xc[i] = node_coords_np[nid, 0]
                yc[i] = node_coords_np[nid, 1]
                zc[i] = node_coords_np[nid, 2]

            Fe = self.compute_element_F(xc, yc, zc)

            for a in range(nNodes):
                for da in range(3):
                    rowA = global_id_np[corner_ids[a]] + da * nNodeGlobal
                    for b in range(nNodes):
                        for vb in range(6):
                            colB = global_id_np[corner_ids[b]] * 6 + vb
                            rows_np[idx] = rowA
                            cols_np[idx] = colB
                            vals_np[idx] = Fe[3 * a + da, 6 * b + vb]
                            idx += 1

        rows_np, cols_np, vals_np = self.impose_anchor_node_dof0_coo_F(
            rows_np, cols_np, vals_np
        )

        return rows_np, cols_np, vals_np

    def impose_anchor_node_dof0_coo(self, rows_np, cols_np, vals_np):
        """Pin displacement DOFs at the anchor node to remove rigid-body modes."""
        anchors = [
            self.anchor_row,
            self.anchor_row + self.nNodeGlobal,
            self.anchor_row + 2 * self.nNodeGlobal,
        ]

        current_rows = rows_np
        current_cols = cols_np
        current_vals = vals_np

        for anchor in anchors:
            for i in range(current_vals.size):
                if current_rows[i] == anchor or current_cols[i] == anchor:
                    current_vals[i] = 0.0

            N = current_rows.size
            new_N = N + 1
            new_rows = np.empty(new_N, dtype=current_rows.dtype)
            new_cols = np.empty(new_N, dtype=current_cols.dtype)
            new_vals = np.empty(new_N, dtype=current_vals.dtype)

            for i in range(N):
                new_rows[i] = current_rows[i]
                new_cols[i] = current_cols[i]
                new_vals[i] = current_vals[i]

            new_rows[N] = anchor
            new_cols[N] = anchor
            new_vals[N] = 1.0

            current_rows = new_rows
            current_cols = new_cols
            current_vals = new_vals

        return current_rows, current_cols, current_vals

    def impose_anchor_node_dof0_coo_F(self, rows_np, cols_np, vals_np):
        """Zero out anchor rows in the coupling matrix F for consistency."""
        anchors = [
            self.anchor_row,
            self.anchor_row + self.nNodeGlobal,
            self.anchor_row + 2 * self.nNodeGlobal,
        ]
        for anchor in anchors:
            for i in range(vals_np.size):
                if rows_np[i] == anchor:
                    vals_np[i] = 0.0

        return rows_np, cols_np, vals_np
