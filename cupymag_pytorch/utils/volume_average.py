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
import torch

from cupymag_pytorch.core.parameters import grid_type, precision

if grid_type == "Hex":
    from cupymag_pytorch.mesh.gridHex import HexGrid
    from cupymag_pytorch.mesh.ShapeHex import (
        gauss_quadrature,
        shape_function_gradients,
        shape_functions,
    )
elif grid_type == "Tet":
    from cupymag_pytorch.mesh.gridTet import TetraGrid
    from cupymag_pytorch.mesh.ShapeTet import (
        gauss_quadrature,
        shape_function_gradients,
        shape_functions,
    )
else:
    raise NotImplementedError(f"Grid type '{grid_type}' is not supported.")

from cupymag_pytorch.utils.backend import DEVICE, to_np
from cupymag_pytorch.utils.precision_select import get_float_type

float_cp = get_float_type(precision, backend="torch")


class VolumeAverage:
    """
    Volume averaging for finite element meshes using Gauss quadrature, plus
    ParaView VTU export with nodal/cell-centered blending.

    Attributes
    ----------
    coords : torch.Tensor (nNodes, 3)
    elements : torch.Tensor (nElems, nNodesPerElem + 1)
    original_global_id : torch.Tensor (nNodes,)
    n_nodes_per_elem : int
    corner_dofs : torch.Tensor (nElems, nNodesPerElem)
    defect_flags : torch.Tensor (nElems,)
    gauss_points, gauss_weights : torch.Tensor
    detJ : torch.Tensor (nElems, nGaussPoints)
    N_gauss_gpu : torch.Tensor (nGaussPoints, nNodesPerElem)
    W_gauss_gpu : torch.Tensor (nGaussPoints,)
    """

    def __init__(self, coords, elements, global_id):
        self.coords = coords
        self.elements = elements

        self.original_global_id = global_id

        self.n_nodes_per_elem = elements.shape[1] - 1
        self.corner_dofs = global_id[elements[:, 0 : self.n_nodes_per_elem]]
        self.defect_flags = elements[:, -1]

        gp, gw = gauss_quadrature()
        self.gauss_points = torch.as_tensor(gp, dtype=float_cp, device=DEVICE)
        self.gauss_weights = torch.as_tensor(gw, dtype=float_cp, device=DEVICE)

        self.detJ = self.compute_detJ_gpu_vectorized()
        self.N_gauss_gpu = self.build_N_gauss()
        self.W_gauss_gpu = self.gauss_weights

    def compute_detJ_gpu_vectorized(self):
        """Compute |det(J)| at every Gauss point for every element."""
        coords = self.coords
        corner_dofs = self.corner_dofs

        n_elems = corner_dofs.shape[0]

        gauss_points = self.gauss_points
        ngp = len(gauss_points)

        corner_coords = coords[corner_dofs]

        detJ_g = torch.zeros((n_elems, ngp), dtype=float_cp, device=DEVICE)

        for igp in range(ngp):
            r, s, t = gauss_points[igp]

            dN = shape_function_gradients(r, s, t)
            dN = dN[: self.n_nodes_per_elem, :]

            J = torch.einsum("enk,nj->ekj", corner_coords, dN)

            detJ = (
                J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1])
                - J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0])
                + J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0])
            )

            detJ = detJ.abs()
            detJ_g[:, igp] = detJ

        return detJ_g

    def build_N_gauss(self):
        """Evaluate shape functions at every Gauss point."""
        gauss_points = self.gauss_points
        n_nodes_per_elem = self.n_nodes_per_elem

        ngp = len(gauss_points)
        N_gauss = torch.zeros((ngp, n_nodes_per_elem), dtype=float_cp, device=DEVICE)

        for igp in range(ngp):
            r, s, t = gauss_points[igp]
            Nvals = shape_functions(r, s, t)
            for b in range(n_nodes_per_elem):
                N_gauss[igp, b] = Nvals[b]

        return N_gauss

    def compute_average_field_gpu(self, m, defect_flag=None):
        """
        Compute the volume-weighted average of a nodal field using Gauss
        quadrature.

        Parameters
        ----------
        m : torch.Tensor (nDOF,) or (nDOF, nComp)
        defect_flag : optional element flags; defective elements excluded.

        Returns
        -------
        torch.Tensor
            Volume-weighted average (scalar for scalar fields, length-nComp
            for vector fields).
        """
        corner_dofs = self.corner_dofs
        N_gauss = self.N_gauss_gpu
        W_gauss = self.W_gauss_gpu
        detJ_g = self.detJ

        is_scalar = m.dim() == 1
        if is_scalar:
            m = m.reshape(-1, 1)

        corner_U = m[corner_dofs]

        MGP = torch.einsum("ebk,gb->egk", corner_U, N_gauss)

        volume_weights = detJ_g * W_gauss
        volume_weights_3 = volume_weights[:, :, None]

        partial_integration = MGP * volume_weights_3

        partial_integration_e = partial_integration.sum(dim=1)

        partial_volume = volume_weights.sum(dim=1)

        if defect_flag is not None:
            df_float = defect_flag.to(float_cp)
            is_not_defect = 1.0 - df_float
            partial_integration_e = partial_integration_e * is_not_defect[:, None]
            partial_volume_summed = (partial_volume * is_not_defect).sum()
        else:
            partial_volume_summed = partial_volume.sum()

        total_integration = partial_integration_e.sum(dim=0)
        total_volume = partial_volume_summed

        if abs(total_volume.item()) < 1e-6:
            print("WARNING: Total volume is very small, results may be unstable")
            total_volume = max(abs(total_volume.item()), 1e-6)

        avg_vec = total_integration / total_volume

        if is_scalar:
            return avg_vec[0]
        else:
            return avg_vec

    def write_to_paraview(self, field_dict, filename, alpha=0.5, eps=1e-14):
        """
        Visualize fields in ParaView with proper handling of periodic
        boundaries.

        Parameters
        ----------
        field_dict : dict
            Field name -> (nDOF,) or (nDOF, nComp) torch tensors.
        filename : str
            Output VTU filename.
        alpha : float
            Blending: 1.0 = purely nodal, 0.0 = purely cell-centered.
        """
        import meshio
        from scipy.sparse import coo_matrix

        coords = to_np(self.coords)
        cells = to_np(self.elements)[:, : self.n_nodes_per_elem].astype(np.int64)
        gid = to_np(self.original_global_id).astype(np.int64)

        n_nodes, n_elems = coords.shape[0], cells.shape[0]
        n_per_elem = self.n_nodes_per_elem

        N_g = to_np(self.N_gauss_gpu)
        W_g = to_np(self.W_gauss_gpu)
        detJ = to_np(self.detJ)
        cDOF = to_np(self.corner_dofs).astype(np.int64)

        # node -> element adjacency (each node belongs to which elements)
        row = cells.reshape(-1)
        col = np.repeat(np.arange(n_elems), n_per_elem)
        data = np.ones_like(row, dtype=np.float64)

        A = coo_matrix((data, (row, col)), shape=(n_nodes, n_elems)).tocsr()
        A_sum = np.asarray(A.sum(axis=1)).ravel()

        uniq, inverse, counts = np.unique(gid, return_inverse=True, return_counts=True)
        dof_inv_cnt = (1.0 / counts)[inverse][:, None]

        point_data = {}
        cell_data = {}

        for name, f in field_dict.items():
            f = to_np(f).astype(np.float64)
            if f.ndim == 1:
                f = f[:, None]
            nC = f.shape[1]

            node_val = f[gid]  # (n_nodes, nC)

            if alpha < 0.999:
                f_e = f[cDOF]  # (e, n, C)
                F_e_g = np.tensordot(f_e, N_g, axes=(1, 1)).transpose(
                    0, 2, 1
                )  # (e, g, C)
                weight = detJ * W_g  # (e, g)

                num = np.einsum("eg,egc->ec", weight, F_e_g)
                den = weight.sum(axis=1, keepdims=True)  # (e, 1)

                elem_val = np.where(den > eps, num / den, f_e.mean(axis=1))

                cc_val = A @ elem_val  # (n, C)
                cc_val = np.where(A_sum[:, None] > 0, cc_val / A_sum[:, None], node_val)

                buf = np.zeros((uniq.size, nC), dtype=cc_val.dtype)
                np.add.at(buf, inverse, cc_val)
                cc_val = buf[inverse] * dof_inv_cnt

                node_val = alpha * node_val + (1.0 - alpha) * cc_val

            if nC == 1:
                point_data[name] = node_val[:, 0]
            else:
                for c in range(nC):
                    point_data[f"{name}_{c + 1}"] = node_val[:, c]

        if getattr(self, "defect_flags", None) is not None:
            cell_data["defect_flag"] = [to_np(self.defect_flags)]

        elem_type = {4: "tetra", 8: "hexahedron"}[n_per_elem]
        meshio.write(
            filename,
            meshio.Mesh(
                points=coords,
                cells=[(elem_type, cells)],
                point_data=point_data,
                cell_data=cell_data,
            ),
        )
        print(f"Field written to {filename}.")
