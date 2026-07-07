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
}


@jitclass(spec)
class AssembleGaussSeidel:
    """
    Finite element assembly class for the Poisson-like systems arising in
    the Gauss-Seidel projection method.

    Assembles the element stiffness matrix K_e[a,b] = (grad N_a, grad N_b)
    and the mass matrix M_e[a,b] = (N_a, N_b); the global A = M + alpha*K
    and F = M are then formed in COO format.
    """

    def __init__(self, node_coords, elements, global_id):
        self.node_coords = node_coords
        self.elements = elements
        self.Nnodes_per_element = self.elements.shape[1] - 1
        self.gauss_points, self.gauss_weights = gauss_quadrature()
        self.global_id = global_id

    def compute_element_stiffness_cpu(self, xc, yc, zc):
        """Element stiffness matrix K_e[a,b] = (grad N_a, grad N_b)."""
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

    def compute_element_mass_cpu(self, xc, yc, zc):
        """Element mass matrix M_e[a,b] = (N_a, N_b)."""
        nNodes = self.Nnodes_per_element
        ngp = len(self.gauss_points)

        M_e = np.zeros((nNodes, nNodes), dtype=np.float64)

        for igp in range(ngp):
            J_ = element_jacobian(xc, yc, zc, igp)

            detJ = np.linalg.det(J_)
            if abs(detJ) < 1e-14:
                return np.zeros((nNodes, nNodes))

            w = self.gauss_weights[igp]

            N_ = get_N(igp)

            for a in range(nNodes):
                for b in range(nNodes):
                    M_e[a, b] += N_[a] * N_[b] * (detJ * w)

        return M_e

    def build_coo_matrices_numba(self):
        """Build both A (= alpha*K + M) and F (= M) in COO format simultaneously."""
        node_coords_np = self.node_coords
        elements_np = self.elements
        global_id_np = self.global_id

        Ne = elements_np.shape[0]
        nnz = Ne * (self.Nnodes_per_element**2)

        rows_np = np.empty(nnz, dtype=int_np)
        cols_np = np.empty(nnz, dtype=int_np)
        vals_K_np = np.zeros(nnz, dtype=np.float64)
        vals_F_np = np.zeros(nnz, dtype=np.float64)

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

            M_e = self.compute_element_mass_cpu(xc, yc, zc)
            K_e = self.compute_element_stiffness_cpu(xc, yc, zc)

            for a in range(self.Nnodes_per_element):
                ra = global_id_np[corner_ids[a]]
                for b in range(self.Nnodes_per_element):
                    cb = global_id_np[corner_ids[b]]
                    rows_np[idx] = ra
                    cols_np[idx] = cb
                    vals_K_np[idx] = K_e[a, b]
                    vals_F_np[idx] = M_e[a, b]
                    idx += 1

        rows_F = rows_np.copy()
        cols_F = cols_np.copy()

        return rows_np, cols_np, vals_K_np, rows_F, cols_F, vals_F_np
