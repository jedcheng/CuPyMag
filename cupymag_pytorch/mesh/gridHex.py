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

from cupymag_pytorch.utils.backend import DEVICE, to_np

int_np = np.int32


class HexGrid:
    """
    A 3D hexahedral finite element mesh generator for cubic domains with
    embedded defect regions.

    Generates structured hexahedral meshes on rectangular domains
    [0, Nx] x [0, Ny] x [0, Nz], distinguishing a centered cubic defect
    region from the surrounding magnetic material.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of elements along x, y, z.
    Ndx, Ndy, Ndz : int
        Dimensions of the centered defect region in x, y, z.
    defect_center : array_like, optional
        Defect center (defaults to domain center).
    """

    def __init__(self, Nx, Ny, Nz, Ndx, Ndy, Ndz, defect_center=None):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Ndx = Ndx
        self.Ndy = Ndy
        self.Ndz = Ndz
        self.defect_center = defect_center
        self.x = np.linspace(0, Nx, Nx + 1)
        self.y = np.linspace(0, Ny, Ny + 1)
        self.z = np.linspace(0, Nz, Nz + 1)

    def std_fem_mesh(self):
        """
        Generate node coordinates and element connectivity.

        Returns
        -------
        node_coords : torch.Tensor (Nnodes, 3)
        elements : torch.Tensor (Nelems, 9)
            elements[e, 0..7] = corner node indices,
            elements[e, 8]    = defect flag in {0, 1}.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        Ndx, Ndy, Ndz = self.Ndx, self.Ndy, self.Ndz

        x, y, z = self.x, self.y, self.z

        # Node coordinates via vectorized meshgrid (built on CPU).
        i3, j3, k3 = np.meshgrid(
            np.arange(Nx + 1), np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij"
        )
        node_coords_np = np.stack(
            (x[i3.ravel()], y[j3.ravel()], z[k3.ravel()]), axis=1
        ).astype(np.float64)

        # Defect boundaries (python floats for the per-element loop).
        if self.defect_center is None:
            cx, cy, cz = Nx / 2.0, Ny / 2.0, Nz / 2.0
        else:
            cx, cy, cz = (float(v) for v in to_np(self.defect_center).ravel())

        x_min, x_max = cx - Ndx / 2.0, cx + Ndx / 2.0
        y_min, y_max = cy - Ndy / 2.0, cy + Ndy / 2.0
        z_min, z_max = cz - Ndz / 2.0, cz + Ndz / 2.0

        Ne = Nx * Ny * Nz
        elements_np = np.zeros((Ne, 9), dtype=int_np)

        elem_idx = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):

                    def node_id(ii, jj, kk):
                        return ii * (Ny + 1) * (Nz + 1) + jj * (Nz + 1) + kk

                    elements_np[elem_idx, 0] = node_id(i, j, k)
                    elements_np[elem_idx, 1] = node_id(i + 1, j, k)
                    elements_np[elem_idx, 2] = node_id(i + 1, j + 1, k)
                    elements_np[elem_idx, 3] = node_id(i, j + 1, k)
                    elements_np[elem_idx, 4] = node_id(i, j, k + 1)
                    elements_np[elem_idx, 5] = node_id(i + 1, j, k + 1)
                    elements_np[elem_idx, 6] = node_id(i + 1, j + 1, k + 1)
                    elements_np[elem_idx, 7] = node_id(i, j + 1, k + 1)

                    xc = i + 0.5
                    yc = j + 0.5
                    zc = k + 0.5

                    is_defect = (
                        (xc >= x_min)
                        and (xc < x_max)
                        and (yc >= y_min)
                        and (yc < y_max)
                        and (zc >= z_min)
                        and (zc < z_max)
                    )
                    elements_np[elem_idx, 8] = int(is_defect)
                    elem_idx += 1

        node_coords = torch.from_numpy(node_coords_np).to(DEVICE)
        elements = torch.from_numpy(elements_np).to(DEVICE)
        return node_coords, elements

    def write_mesh_to_paraview(self, filename="example_3d_mesh.vtu"):
        """Write the mesh to a .vtu file for ParaView visualization."""
        import meshio

        node_coords, elements = self.std_fem_mesh()
        points = to_np(node_coords)
        elements_np = to_np(elements)

        connectivity = elements_np[:, :8]
        defect_flags = elements_np[:, 8]

        mesh = meshio.Mesh(
            points=points,
            cells=[("hexahedron", connectivity)],
            cell_data={"defect_or_mag": [defect_flags]},
        )

        meshio.write(filename, mesh)
        print(f"VTU file successfully written: {filename}")

    def build_periodic_node_map(self, node_coords, tol=1e-12):
        """
        Identify which nodes are 'the same' under x/y/z periodicity and
        unify them. Returns global_id[node] = compressed DOF index.
        """
        x = self.x
        y = self.y
        z = self.z

        coords = to_np(node_coords)

        def wrap_dim(val, v0, v1):
            if abs(val - v1) < tol:
                return float(v0)
            return float(val)

        canon_dict = {}
        global_id = np.zeros(coords.shape[0], dtype=int_np)
        next_id = 0

        for n in range(coords.shape[0]):
            xx = wrap_dim(coords[n, 0], x[0], x[-1])
            yy = wrap_dim(coords[n, 1], y[0], y[-1])
            zz = wrap_dim(coords[n, 2], z[0], z[-1])
            key = (round(xx, 12), round(yy, 12), round(zz, 12))
            if key not in canon_dict:
                canon_dict[key] = next_id
                next_id += 1
            global_id[n] = canon_dict[key]

        print(
            f"Periodic node mapping: {coords.shape[0]} nodes -> {len(np.unique(global_id))} unique DOFs"
        )
        return global_id
