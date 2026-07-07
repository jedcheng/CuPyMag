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

from io import StringIO

import numpy as np
import pandas as pd
import torch

from cupymag_pytorch.utils.backend import DEVICE, to_np
from cupymag_pytorch.utils.defect_shapes import *  # noqa: F401,F403

int_np = np.int32


class TetraGrid:
    """
    A tetrahedral finite element mesh processor for 3D domains with
    embedded defect regions.

    Reads tetrahedral meshes from Nastran (.nas) format files and
    classifies elements inside a specified defect region.

    Parameters
    ----------
    nas_filename : str
        Path to the Nastran (.nas) mesh file.
    defect_shape : callable
        Function defining the defect region geometry.
    defect_center : array_like, optional
        Coordinates of the defect center. Defaults to the mesh bounding-box
        center.
    """

    def __init__(self, nas_filename, defect_shape, defect_center=None):
        self.nas_filename = nas_filename
        self.defect_shape = defect_shape
        self.defect_center = defect_center

        self.node_ids, self.node_coords, self.element_ids, self.element_nodes = (
            self.read_nas_file_pd()
        )

        if self.defect_center is None:
            min_coords = np.min(self.node_coords, axis=0)
            max_coords = np.max(self.node_coords, axis=0)
            self.defect_center = (max_coords + min_coords) / 2

        self.defect_center = torch.as_tensor(
            np.asarray(self.defect_center), dtype=torch.float64, device=DEVICE
        )

    def read_nas_file_pd(self):
        """
        Read a NAS file and extract node coordinates and element
        connectivity using pandas (whitespace-delimited parsing).

        Returns
        -------
        node_ids : np.ndarray
        node_coords : np.ndarray (n_nodes, 3)
        element_ids : np.ndarray
        element_nodes_idx : np.ndarray (n_elements, 4)
        """
        grids, tetras = [], []
        with open(self.nas_filename, "r") as fh:
            for raw in fh:
                if raw.startswith("$") or raw.isspace():
                    continue
                if raw.lstrip().startswith("ENDDATA"):
                    break

                line = raw.rstrip()

                if line.startswith("GRID"):
                    grids.append(line)
                elif line.startswith("CTETRA"):
                    tetras.append(line)

        if not grids or not tetras:
            raise ValueError("File contains no GRID or CTETRA cards")

        csv_opts = dict(
            header=None, engine="c", sep=",", skipinitialspace=True, dtype=str
        )

        df_grid = pd.read_csv(
            StringIO("\n".join(grids)),
            names=["GRID", "ID", "CP", "X", "Y", "Z"],
            **csv_opts,
        )[["ID", "X", "Y", "Z"]]
        df_grid = df_grid[["ID", "X", "Y", "Z"]]

        df_tet = pd.read_csv(
            StringIO("\n".join(tetras)),
            names=["CTETRA", "EID", "PID", "N1", "N2", "N3", "N4"],
            **csv_opts,
        )[["EID", "N1", "N2", "N3", "N4"]]
        df_tet = df_tet[["EID", "N1", "N2", "N3", "N4"]]

        node_ids = df_grid["ID"].astype(np.int32).to_numpy(copy=False)
        node_xyz = df_grid[["X", "Y", "Z"]].astype(np.float64).to_numpy(copy=False)
        elem_ids = df_tet["EID"].astype(np.int32).to_numpy(copy=False)
        elem_nodes = (
            df_tet[["N1", "N2", "N3", "N4"]].astype(np.int32).to_numpy(copy=False)
        )

        sort_idx = np.argsort(node_ids)
        sorted_node_ids = node_ids[sort_idx]

        elem_nodes_idx = sort_idx[
            np.searchsorted(sorted_node_ids, elem_nodes, side="left")
        ]

        return node_ids, node_xyz, elem_ids, elem_nodes_idx

    def std_fem_mesh(self):
        """
        Return the tetrahedral mesh in standard FEM format.

        Returns
        -------
        node_coords : torch.Tensor (n_nodes, 3)
        elements : torch.Tensor (n_elements, 5)
            First 4 columns are node indices; the last column is the defect
            flag (0 or 1).
        """
        node_coords = torch.as_tensor(
            self.node_coords, dtype=torch.float64, device=DEVICE
        )

        n_elements = len(self.element_nodes)
        elements = torch.zeros((n_elements, 5), dtype=torch.int32, device=DEVICE)
        elements[:, :4] = torch.as_tensor(
            self.element_nodes, dtype=torch.int32, device=DEVICE
        )

        centers = node_coords[elements[:, :4]].mean(dim=1)
        rel_coords = centers - self.defect_center

        mask = self.defect_shape(rel_coords)
        elements[:, 4] = mask.to(torch.int32)

        return node_coords, elements

    def build_periodic_node_map(self, node_coords, tol=1e-12):
        """
        Identify which nodes are 'the same' under x/y/z periodicity and
        unify them. Returns global_id[node] = compressed DOF index.
        """
        coords = to_np(node_coords)

        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)

        def wrap_dim(val, v0, v1):
            if abs(val - v1) < tol:
                return float(v0)
            return float(val)

        canon_dict = {}
        global_id = np.zeros(coords.shape[0], dtype=int_np)
        next_id = 0

        for n in range(coords.shape[0]):
            xx = wrap_dim(coords[n, 0], min_coords[0], max_coords[0])
            yy = wrap_dim(coords[n, 1], min_coords[1], max_coords[1])
            zz = wrap_dim(coords[n, 2], min_coords[2], max_coords[2])
            key = (round(xx, 12), round(yy, 12), round(zz, 12))
            if key not in canon_dict:
                canon_dict[key] = next_id
                next_id += 1
            global_id[n] = canon_dict[key]

        print(
            f"Periodic node mapping: {coords.shape[0]} nodes -> {len(np.unique(global_id))} unique DOFs"
        )
        return global_id

    def write_mesh_to_paraview(self, filename="tetrahedral_mesh.vtu"):
        """Write the mesh to a .vtu file for ParaView visualization."""
        import meshio

        node_coords, elements = self.std_fem_mesh()
        node_coords_np = to_np(node_coords)
        elements_np = to_np(elements)

        tetra_cells = elements_np[:, :4].astype(int_np)
        defect_flags = elements_np[:, 4].astype(int_np)

        mesh = meshio.Mesh(
            points=node_coords_np,
            cells=[("tetra", tetra_cells)],
            cell_data={"defect_flag": [defect_flags]},
        )

        meshio.write(filename, mesh)
        print(f"VTU file successfully written: {filename}")
