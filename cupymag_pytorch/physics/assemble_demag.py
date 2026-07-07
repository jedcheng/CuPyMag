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

from cupymag_pytorch.core.parameters import grid_type
from cupymag_pytorch.utils.numba_shim import float64, int32, int64, jitclass

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
}


@jitclass(spec)
class AssembleDemag:
    """
    Finite element assembly class for magnetostatic equilibrium.

    Assembles stiffness matrices and derivative operators for solving
    Laplace's equation for the demagnetization potential. The CPU
    assembly is JIT-compiled with Numba when available (see
    ``cupymag_pytorch.utils.numba_shim``); otherwise it runs as plain
    NumPy/Python.

    Attributes
    ----------
    node_coords : (n_nodes, 3) array
    elements : (n_elems, n_nodes_per_elem + 1) int array
    Nnodes_per_element : int
    gauss_points : (n_gp, 3) array
    gauss_weights : (n_gp,) array
    global_id : (n_nodes,) int array (periodic DOF map)
    anchor_row : int
        DOF index pinned to zero to remove the Laplacian nullspace.
    """

    def __init__(self, node_coords, elements, global_id=None):
        self.node_coords = node_coords
        self.elements = elements
        self.Nnodes_per_element = self.elements.shape[1] - 1
        self.gauss_points, self.gauss_weights = gauss_quadrature()
        self.global_id = global_id
        self.anchor_row = 0

    def compute_element_stiffness_cpu(self, xc, yc, zc):
        """Assemble element stiffness matrix K_e[a,b] = (grad N_a, grad N_b)."""
        nNodes = self.Nnodes_per_element
        ngp = len(self.gauss_points)

        K_e = np.zeros((nNodes, nNodes), dtype=np.float64)

        for igp in range(ngp):
            J_ = element_jacobian(xc, yc, zc, igp)
            detJ = np.linalg.det(J_)
            if abs(detJ) < 1e-14:
                return np.zeros((nNodes, nNodes))
            J_inv = np.linalg.inv(J_)

            w = self.gauss_weights[igp]
            dN_rst = get_dN(igp)
            for a in range(nNodes):
                gx_a = (
                    J_inv[0, 0] * dN_rst[a, 0]
                    + J_inv[0, 1] * dN_rst[a, 1]
                    + J_inv[0, 2] * dN_rst[a, 2]
                )
                gy_a = (
                    J_inv[1, 0] * dN_rst[a, 0]
                    + J_inv[1, 1] * dN_rst[a, 1]
                    + J_inv[1, 2] * dN_rst[a, 2]
                )
                gz_a = (
                    J_inv[2, 0] * dN_rst[a, 0]
                    + J_inv[2, 1] * dN_rst[a, 1]
                    + J_inv[2, 2] * dN_rst[a, 2]
                )
                for b in range(nNodes):
                    gx_b = (
                        J_inv[0, 0] * dN_rst[b, 0]
                        + J_inv[0, 1] * dN_rst[b, 1]
                        + J_inv[0, 2] * dN_rst[b, 2]
                    )
                    gy_b = (
                        J_inv[1, 0] * dN_rst[b, 0]
                        + J_inv[1, 1] * dN_rst[b, 1]
                        + J_inv[1, 2] * dN_rst[b, 2]
                    )
                    gz_b = (
                        J_inv[2, 0] * dN_rst[b, 0]
                        + J_inv[2, 1] * dN_rst[b, 1]
                        + J_inv[2, 2] * dN_rst[b, 2]
                    )
                    dot_ab = gx_a * gx_b + gy_a * gy_b + gz_a * gz_b
                    K_e[a, b] += dot_ab * (detJ * w)
        return K_e

    def compute_element_F_cpu(self, xc, yc, zc):
        """Assemble element F matrices Fex[a,b] = (N_a, dN_b/dx), etc."""
        nNodes = self.Nnodes_per_element
        ngp = len(self.gauss_points)

        Fex = np.zeros((nNodes, nNodes), dtype=np.float64)
        Fey = np.zeros((nNodes, nNodes), dtype=np.float64)
        Fez = np.zeros((nNodes, nNodes), dtype=np.float64)

        for igp in range(ngp):
            w = self.gauss_weights[igp]

            J_ = element_jacobian(xc, yc, zc, igp)

            detJ = np.linalg.det(J_)
            if abs(detJ) < 1e-14:
                Z = np.zeros((nNodes, nNodes))
                return Z, Z, Z
            J_inv = np.linalg.inv(J_)

            dN_rst = get_dN(igp)
            N_rst = get_N(igp)
            grad_xyz = np.zeros((nNodes, 3))

            for a in range(nNodes):
                grad_xyz[a, 0] = (
                    J_inv[0, 0] * dN_rst[a, 0]
                    + J_inv[0, 1] * dN_rst[a, 1]
                    + J_inv[0, 2] * dN_rst[a, 2]
                )
                grad_xyz[a, 1] = (
                    J_inv[1, 0] * dN_rst[a, 0]
                    + J_inv[1, 1] * dN_rst[a, 1]
                    + J_inv[1, 2] * dN_rst[a, 2]
                )
                grad_xyz[a, 2] = (
                    J_inv[2, 0] * dN_rst[a, 0]
                    + J_inv[2, 1] * dN_rst[a, 1]
                    + J_inv[2, 2] * dN_rst[a, 2]
                )

            for a in range(nNodes):
                for b in range(nNodes):
                    Fex[a, b] += N_rst[a] * grad_xyz[b, 0] * (detJ * w)
                    Fey[a, b] += N_rst[a] * grad_xyz[b, 1] * (detJ * w)
                    Fez[a, b] += N_rst[a] * grad_xyz[b, 2] * (detJ * w)
        return Fex, Fey, Fez

    def build_coo_matrix_A_numba(self):
        """Build the global stiffness matrix in COO format (one DOF anchored to 0)."""
        node_coords_np = self.node_coords
        elements_np = self.elements
        global_id_np = self.global_id

        Ne = elements_np.shape[0]

        nnz = Ne * (self.Nnodes_per_element) ** 2

        rows_np = np.empty(nnz, dtype=int_np)
        cols_np = np.empty(nnz, dtype=int_np)
        vals_np = np.empty(nnz, dtype=np.float64)

        idx = 0
        for e in range(Ne):
            corner_ids = elements_np[e, 0 : self.Nnodes_per_element]

            xc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            yc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            zc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            for i in range(self.Nnodes_per_element):
                nid = corner_ids[i]
                xc[i] = node_coords_np[nid, 0]
                yc[i] = node_coords_np[nid, 1]
                zc[i] = node_coords_np[nid, 2]

            K_e = self.compute_element_stiffness_cpu(xc, yc, zc)

            for a in range(self.Nnodes_per_element):
                ra = global_id_np[corner_ids[a]]
                for b in range(self.Nnodes_per_element):
                    cb = global_id_np[corner_ids[b]]
                    rows_np[idx] = ra
                    cols_np[idx] = cb
                    vals_np[idx] = K_e[a, b]
                    idx += 1

        rows_np, cols_np, vals_np = self.impose_anchor_node_dof0_coo(
            rows_np, cols_np, vals_np
        )
        return rows_np, cols_np, vals_np

    def build_coo_matrices_F_numba(self):
        """Build the global F (derivative) matrices in COO format."""
        node_coords_np = self.node_coords
        elements_np = self.elements
        global_id_np = self.global_id

        Ne = elements_np.shape[0]

        nnz = Ne * (self.Nnodes_per_element) ** 2

        Fx_rows = np.empty(nnz, dtype=int_np)
        Fx_cols = np.empty(nnz, dtype=int_np)
        Fx_vals = np.zeros(nnz, dtype=np.float64)

        Fy_rows = np.empty(nnz, dtype=int_np)
        Fy_cols = np.empty(nnz, dtype=int_np)
        Fy_vals = np.zeros(nnz, dtype=np.float64)

        Fz_rows = np.empty(nnz, dtype=int_np)
        Fz_cols = np.empty(nnz, dtype=int_np)
        Fz_vals = np.zeros(nnz, dtype=np.float64)

        idx = 0
        for e in range(Ne):
            corner_ids = elements_np[e, 0 : self.Nnodes_per_element]

            xc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            yc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            zc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            for i in range(self.Nnodes_per_element):
                nid = corner_ids[i]
                xc[i] = node_coords_np[nid, 0]
                yc[i] = node_coords_np[nid, 1]
                zc[i] = node_coords_np[nid, 2]

            Fex, Fey, Fez = self.compute_element_F_cpu(xc, yc, zc)

            for a in range(self.Nnodes_per_element):
                ra = global_id_np[corner_ids[a]]
                for b in range(self.Nnodes_per_element):
                    cb = global_id_np[corner_ids[b]]
                    Fx_rows[idx] = ra
                    Fx_cols[idx] = cb
                    Fx_vals[idx] = Fex[a, b]

                    Fy_rows[idx] = ra
                    Fy_cols[idx] = cb
                    Fy_vals[idx] = Fey[a, b]

                    Fz_rows[idx] = ra
                    Fz_cols[idx] = cb
                    Fz_vals[idx] = Fez[a, b]

                    idx += 1

        return (
            Fx_rows,
            Fx_cols,
            Fx_vals,
            Fy_rows,
            Fy_cols,
            Fy_vals,
            Fz_rows,
            Fz_cols,
            Fz_vals,
        )

    def impose_anchor_node_dof0_coo(self, rows_np, cols_np, vals_np):
        """Pin the potential at the anchor DOF to zero (remove Laplacian nullspace)."""
        anchor = self.anchor_row

        for i in range(vals_np.size):
            if rows_np[i] == anchor or cols_np[i] == anchor:
                vals_np[i] = 0.0

        N = rows_np.size

        new_N = N + 1
        new_rows = np.empty(new_N, dtype=rows_np.dtype)
        new_cols = np.empty(new_N, dtype=cols_np.dtype)
        new_vals = np.empty(new_N, dtype=vals_np.dtype)

        for i in range(N):
            new_rows[i] = rows_np[i]
            new_cols[i] = cols_np[i]
            new_vals[i] = vals_np[i]

        new_rows[N] = anchor
        new_cols[N] = anchor
        new_vals[N] = 1.0

        return new_rows, new_cols, new_vals

    def impose_anchor_node_dof0_coo_F(self, rows_np, cols_np, vals_np):
        """Zero out the anchor row in the F matrix so rhs[anchor] = 0."""
        anchor = self.anchor_row
        for i in range(vals_np.size):
            if rows_np[i] == anchor:
                vals_np[i] = 0.0

        return rows_np, cols_np, vals_np
