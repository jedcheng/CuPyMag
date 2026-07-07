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

from cupymag_pytorch.core.parameters import (
    ME,
    alpha,
    alpha_damp,
    c11,
    c12,
    c44,
    lambda100,
    lambda111,
)
from cupymag_pytorch.mesh.setup_FEM_mesh import FEMMesh
from cupymag_pytorch.physics.assemble_demag import AssembleDemag
from cupymag_pytorch.physics.assemble_elasticity import AssembleElasticity
from cupymag_pytorch.physics.assemble_Gauss_Seidel import AssembleGaussSeidel
from cupymag_pytorch.utils.compute_derivatives import ComputeDerivatives
from cupymag_pytorch.utils.final_assembly import *
from cupymag_pytorch.utils.volume_average import VolumeAverage


class SimulationOperators(FEMMesh):
    """
    Assemble and return all the operators used throughout the simulation:
      1. Magnetostatic (demagnetization) equilibrium system
      2. Gauss-Seidel projection method (GSPM) systems
      3. Mechanical (elastic) equilibrium system
      4. Spatial derivative operators
      5. Volume average operators
    """

    def __init__(self):
        super().__init__()
        self.alpha1 = alpha
        self.alpha2 = alpha * alpha_damp

    def csr_demag_and_deriv(self):
        """
        Assemble the demagnetization stiffness matrix and the Fx/Fy/Fz
        mass/derivative matrices as device SparseMat objects.
        """
        DemagAssembler = AssembleDemag(
            self.node_coords_np, self.elements_np, self.global_id_np
        )
        rows_np, cols_np, vals_np = DemagAssembler.build_coo_matrix_A_numba()
        A_demag = assemble_stiffness_matrix(rows_np, cols_np, vals_np, nDOFx=self.n_dof)

        (Fx_r, Fx_c, Fx_v, Fy_r, Fy_c, Fy_v, Fz_r, Fz_c, Fz_v) = (
            DemagAssembler.build_coo_matrices_F_numba()
        )
        Fx = assemble_mass_matrix(Fx_r, Fx_c, Fx_v, nDOFx=self.n_dof)
        Fy = assemble_mass_matrix(Fy_r, Fy_c, Fy_v, nDOFx=self.n_dof)
        Fz = assemble_mass_matrix(Fz_r, Fz_c, Fz_v, nDOFx=self.n_dof)

        Deriv = ComputeDerivatives(Fx, Fy, Fz)

        return A_demag, Fx, Fy, Fz, Deriv

    def csr_GS(self):
        """
        Assemble the GSPM systems:
          A1 = alpha1 * K + M,  A2 = alpha2 * K + M,  F = M.
        """
        GSAssembler = AssembleGaussSeidel(
            self.node_coords_np, self.elements_np, self.global_id_np
        )
        rows_np, cols_np, vals_np, Fx_r, Fx_c, Fx_v = (
            GSAssembler.build_coo_matrices_numba()
        )

        A1_GS = assemble_stiffness_matrix(
            rows_np,
            cols_np,
            vals_np * self.alpha1 + Fx_v,
            nDOFx=self.n_dof,
            defect_dofs=self.DefDOF,
        )
        A2_GS = assemble_stiffness_matrix(
            rows_np,
            cols_np,
            vals_np * self.alpha2 + Fx_v,
            nDOFx=self.n_dof,
            defect_dofs=self.DefDOF,
        )
        F_GS = assemble_mass_matrix(
            Fx_r, Fx_c, Fx_v, nDOFx=self.n_dof, defect_dofs=self.DefDOF
        )

        return A1_GS, A2_GS, F_GS

    def csr_elasticity(self):
        """
        Assemble the linear-elasticity stiffness and coupling matrices.
        Returns (None, None) when magnetoelastic coupling is disabled.
        """
        if not ME:
            return None, None

        ElasticityAssembler = AssembleElasticity(
            self.node_coords_np,
            self.elements_np,
            self.global_id_np,
            c11,
            c12,
            c44,
            lambda100,
            lambda111,
        )
        rows_np, cols_np, vals_np = ElasticityAssembler.build_coo_matrix_A_numba()
        Fx_r, Fx_c, Fx_v = ElasticityAssembler.build_coo_matrices_F_numba()
        A_el = assemble_stiffness_matrix(
            rows_np, cols_np, vals_np, nDOFx=self.n_dof * 3
        )
        F_el = assemble_mass_matrix(
            Fx_r, Fx_c, Fx_v, nDOFx=self.n_dof * 3, nDOFy=self.n_dof * 6
        )

        return A_el, F_el

    def volume_average(self):
        """Precompute Jacobians and shape functions for volume averaging."""
        Avg = VolumeAverage(self.node_coords_pt, self.elements_pt, self.global_id_pt)
        return Avg
