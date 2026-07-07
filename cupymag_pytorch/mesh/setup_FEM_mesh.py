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

from cupymag_pytorch.core.parameters import *
from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.defect_shapes import get_defect_shape_function
from cupymag_pytorch.utils.final_assembly import extract_defect_dofs

if grid_type == "Hex":
    from cupymag_pytorch.mesh.gridHex import HexGrid
elif grid_type == "Tet":
    from cupymag_pytorch.mesh.gridTet import TetraGrid
else:
    raise NotImplementedError(f"Grid type '{grid_type}' is not supported.")


class FEMMesh:
    """
    Finite Element Method mesh handler.

    Generates and stores the mesh (node coordinates, element connectivity,
    periodic DOF map) in both NumPy (for CPU assembly) and torch (device)
    forms.
    """

    DefDOF = None
    _mesh_computed = False

    def __init__(self):
        self.grid_type = grid_type
        self.fem_grid = None

        self.node_coords_np = None
        self.node_coords_pt = None
        self.elements_np = None
        self.elements_pt = None
        self.global_id_np = None
        self.global_id_pt = None

        self.n_dof = None
        self.n_nodes = None
        self.n_elements = None

        if not FEMMesh._mesh_computed:
            self.generate_mesh()
            FEMMesh._mesh_computed = True

            if FEMMesh.DefDOF is None:
                FEMMesh.DefDOF = extract_defect_dofs(
                    self.elements_pt, self.global_id_pt
                )

    def generate_mesh(self):
        """Generate the finite element mesh based on grid type."""
        defect_center = globals().get("defect_center", None)

        if self.grid_type == "Hex":
            self.fem_grid = HexGrid(Nx, Ny, Nz, Ndx, Ndy, Ndz, defect_center)
        elif self.grid_type == "Tet":
            shape_fn = get_defect_shape_function(defect_shape, shape_params)
            self.fem_grid = TetraGrid(mesh_file, shape_fn, defect_center)

        # device tensors
        self.node_coords_pt, self.elements_pt = self.fem_grid.std_fem_mesh()

        # numpy copies for the CPU assembly
        self.node_coords_np = (
            self.node_coords_pt.detach().cpu().numpy().astype(np.float64)
        )
        self.elements_np = self.elements_pt.detach().cpu().numpy().astype(np.int32)

        self.global_id_np = self.fem_grid.build_periodic_node_map(self.node_coords_np)
        self.global_id_pt = torch.as_tensor(self.global_id_np, device=DEVICE)

        self.n_dof = len(np.unique(self.global_id_np))
        self.n_nodes = self.node_coords_pt.shape[0]
        self.n_elements = self.elements_pt.shape[0]

        return self

    def get_mesh_data(self):
        """Get mesh data for computation."""
        return {
            "node_coords_np": self.node_coords_np,
            "node_coords_pt": self.node_coords_pt,
            "elements_np": self.elements_np,
            "elements_pt": self.elements_pt,
            "global_id_np": self.global_id_np,
            "global_id_pt": self.global_id_pt,
            "n_dof": self.n_dof,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
        }
