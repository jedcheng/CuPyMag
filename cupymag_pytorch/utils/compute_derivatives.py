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

import torch


class ComputeDerivatives:
    """
    Computes spatial derivatives and derived fields (demagnetization field,
    strain) in the FEM setting.

    Attributes
    ----------
    Dx, Dy, Dz : SparseMat
        Derivative operators w.r.t. x, y, z (= -Fx, -Fy, -Fz in the
        original notation; here they hold the F operators directly).
    """

    def __init__(self, Fx, Fy, Fz):
        self.Dx = Fx
        self.Dy = Fy
        self.Dz = Fz

    def compute_Hd_from_U(self, U):
        """
        Compute the demagnetization field H_d from the potential U.

        Returns
        -------
        (Hd1, Hd2, Hd3) : tuple of torch.Tensor (each length n)
        """
        Hd1 = -(self.Dx @ U)
        Hd2 = -(self.Dy @ U)
        Hd3 = -(self.Dz @ U)

        return Hd1, Hd2, Hd3

    def compute_E_from_u(self, n, u, R=None):
        """
        Compute the Voigt strain from displacement ``u``, optionally
        rotating the derivative operators by ``R`` without forming new
        n x n matrices.

        Parameters
        ----------
        n : int
            Number of scalar DOFs per coordinate (nDOF).
        u : torch.Tensor (3n,)
            Concatenation [u_x; u_y; u_z] in the x-basis.
        R : torch.Tensor (3, 3) or None
            Rotation from x' -> x. If None, no rotation is applied.

        Returns
        -------
        torch.Tensor (n, 6)
            Voigt strains [E11, E22, E33, E12, E23, E13] per DOF.
        """
        Dx = self.Dx
        Dy = self.Dy
        Dz = self.Dz

        u_x = u[0:n]
        u_y = u[n : 2 * n]
        u_z = u[2 * n : 3 * n]

        if R is None:
            E11 = Dx @ u_x
            E22 = Dy @ u_y
            E33 = Dz @ u_z

            E12 = Dy @ u_x + Dx @ u_y
            E23 = Dz @ u_y + Dy @ u_z
            E13 = Dz @ u_x + Dx @ u_z

        else:
            dxx = Dx @ u_x
            dyx = Dy @ u_x
            dzx = Dz @ u_x

            dxy = Dx @ u_y
            dyy = Dy @ u_y
            dzy = Dz @ u_y

            dxz = Dx @ u_z
            dyz = Dy @ u_z
            dzz = Dz @ u_z

            E11 = (R[0, 0] * dxx) + (R[1, 0] * dyx) + (R[2, 0] * dzx)

            E22 = (R[0, 1] * dxy) + (R[1, 1] * dyy) + (R[2, 1] * dzy)

            E33 = (R[0, 2] * dxz) + (R[1, 2] * dyz) + (R[2, 2] * dzz)

            tmp1 = (R[0, 1] * dxx) + (R[1, 1] * dyx) + (R[2, 1] * dzx)
            tmp2 = (R[0, 0] * dxy) + (R[1, 0] * dyy) + (R[2, 0] * dzy)
            E12 = tmp1 + tmp2

            tmp3 = (R[0, 2] * dxy) + (R[1, 2] * dyy) + (R[2, 2] * dzy)
            tmp4 = (R[0, 1] * dxz) + (R[1, 1] * dyz) + (R[2, 1] * dzz)
            E23 = tmp3 + tmp4

            tmp5 = (R[0, 2] * dxx) + (R[1, 2] * dyx) + (R[2, 2] * dzx)
            tmp6 = (R[0, 0] * dxz) + (R[1, 0] * dyz) + (R[2, 0] * dzz)
            E13 = tmp5 + tmp6

        return torch.stack([E11, E22, E33, E12, E23, E13], dim=-1)
