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

import os
import time

import numpy as np
import torch

from cupymag_pytorch.core.parameters import *
from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.precision_select import get_float_type

float_cp = get_float_type(precision, backend="torch")

from cupymag_pytorch.core.assemble_simulation_operators import SimulationOperators
from cupymag_pytorch.solvers.linear_solvers import solve_cg
from cupymag_pytorch.utils.final_assembly import *
from cupymag_pytorch.utils.magnetization_io import *
from cupymag_pytorch.utils.print_logo import print_logo
from cupymag_pytorch.utils.print_system_info import print_system_info_summary
from cupymag_pytorch.utils.rot_111_matrices import get_M_matrix, get_R_matrix
from cupymag_pytorch.utils.sigma_matrices import get_Ebar_sigma


def main():
    print_logo()
    print_system_info_summary(config_path)

    global Hext1, Hext2, Hext3
    global Htilde1, Htilde2, Htilde3, E, R, M, m100
    global elasC1, elasC2, elasC3
    if ME == False:
        elasC1 = 0.0
        elasC2 = 0.0
        elasC3 = 0.0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(
            f"Directory {output_dir} did not exist previously, create directory {output_dir}."
        )

    micromagnetics_start_time = time.time()

    # Rotation matrix [100] -> [111]: y = R x, x in [100], y in [111]
    if rot111:
        R = get_R_matrix(backend="torch", dtype=float_cp)
        M = get_M_matrix(backend="torch", dtype=float_cp)
        M_inv_T = torch.linalg.inv(M).T

    # Strains from the external stress
    E_bar_sigma = get_Ebar_sigma()

    # Assemble system operators
    SimOp = SimulationOperators()
    print(
        f"Initialized {grid_type} FEM mesh with #nodes={SimOp.n_nodes}, "
        f"#elements={SimOp.n_elements}, "
        f"#defect elements={int(torch.sum(SimOp.elements_pt[:, -1]).item())}."
    )
    A_demag, Fx, Fy, Fz, Deriv = SimOp.csr_demag_and_deriv()
    A1, A2, F1 = SimOp.csr_GS()
    A_el, F_el = SimOp.csr_elasticity()
    Avg = SimOp.volume_average()

    # Optional Jacobi (inverse-diagonal) preconditioners.
    if cg_preconditioner == "jacobi":
        Mj_demag = 1.0 / A_demag.diagonal()
        Mj_A1 = 1.0 / A1.diagonal()
        Mj_A2 = 1.0 / A2.diagonal()
        Mj_el = 1.0 / A_el.diagonal() if ME else None
    else:
        Mj_demag = Mj_A1 = Mj_A2 = Mj_el = None

    # Initialize magnetization
    DefDOF = SimOp.DefDOF
    nDOF = SimOp.n_dof
    m = initialize_m(restart_m, restart, initial_magnetization, nDOF, DefDOF, float_cp)

    avg_m = Avg.compute_average_field_gpu(m)
    m_tilde = m - avg_m

    # CG solver parameter
    use_init = False

    # Gauss-Seidel step counters
    nstep = 0
    count = 0
    sumsq = float("inf")

    if ME == True:
        llg_min_step = 10
    else:
        llg_min_step = 100

    LLG_accuracy = nDOF * LLG_accuracy_factor

    m_prev = m.clone()

    U = None
    u = None
    g1n = None
    g2n = None
    g3n = None
    g1star = None
    g2star = None
    m1starstar = None
    m2starstar = None
    m3starstar = None

    hyst_file = open(os.path.join(output_dir, "hysteresis.txt"), "w", buffering=1)
    hyst_file.write("Hext1\tHext2\tHext3\tavg_m1\tavg_m2\tavg_m3\n")

    LLG_start_time = time.time()

    print(
        f"FEM system set up done using {LLG_start_time - micromagnetics_start_time:.2f} s, "
        f"begin LLG iterations."
    )
    while (avg_m @ stop_direction).item() > stop_value:
        nstep += 1
        if rot111 == True:
            m100 = m @ R

        if ME == True:
            if rot111 == True:
                E0 = build_E0_from_m(lambda100, lambda111, m100)
                E0 = E0.reshape((nDOF, 6))
                E0 = E0 @ M.T
                E0 = E0.reshape(6 * nDOF)
            else:
                E0 = build_E0_from_m(lambda100, lambda111, m)

            b_el = F_el @ E0
            u = solve_cg(
                A_el,
                b_el,
                x0=u,
                M=Mj_el,
                tol=tol,
                maxiter=maxiter,
                system="elasticity",
                use_init=use_init,
            check_every=cg_check_every,
            )

            if rot111:
                E = Deriv.compute_E_from_u(nDOF, u, R)
            else:
                E = Deriv.compute_E_from_u(nDOF, u)

            # Remove the volume integral of E (result in E_tilde)
            E = E - Avg.compute_average_field_gpu(E)

            # Compute E_bar
            if rot111 == False:
                E_bar11 = 1.5 * lambda100 * (avg_m[0] ** 2 - 1 / 3) + E_bar_sigma[0]
                E_bar22 = 1.5 * lambda100 * (avg_m[1] ** 2 - 1 / 3) + E_bar_sigma[1]
                E_bar33 = 1.5 * lambda100 * (avg_m[2] ** 2 - 1 / 3) + E_bar_sigma[2]
                E_bar12 = 3.0 * lambda111 * avg_m[0] * avg_m[1] + E_bar_sigma[3]
                E_bar23 = 3.0 * lambda111 * avg_m[1] * avg_m[2] + E_bar_sigma[4]
                E_bar13 = 3.0 * lambda111 * avg_m[0] * avg_m[2] + E_bar_sigma[5]
            else:
                avg_m100 = Avg.compute_average_field_gpu(m100)
                E_bar11 = 1.5 * lambda100 * (avg_m100[0] ** 2 - 1 / 3) + E_bar_sigma[0]
                E_bar22 = 1.5 * lambda100 * (avg_m100[1] ** 2 - 1 / 3) + E_bar_sigma[1]
                E_bar33 = 1.5 * lambda100 * (avg_m100[2] ** 2 - 1 / 3) + E_bar_sigma[2]
                E_bar12 = 3.0 * lambda111 * avg_m100[0] * avg_m100[1] + E_bar_sigma[3]
                E_bar23 = 3.0 * lambda111 * avg_m100[1] * avg_m100[2] + E_bar_sigma[4]
                E_bar13 = 3.0 * lambda111 * avg_m100[0] * avg_m100[2] + E_bar_sigma[5]
                E_bar = torch.stack(
                    [E_bar11, E_bar22, E_bar33, E_bar12, E_bar23, E_bar13]
                )
                E_bar11, E_bar22, E_bar33, E_bar12, E_bar23, E_bar13 = M @ E_bar

            # Compute the "true" total strain
            E[:, 0] = E[:, 0] + E_bar11
            E[:, 1] = E[:, 1] + E_bar22
            E[:, 2] = E[:, 2] + E_bar33
            E[:, 3] = E[:, 3] + E_bar12
            E[:, 4] = E[:, 4] + E_bar23
            E[:, 5] = E[:, 5] + E_bar13

            if rot111 == True:
                E = E @ M_inv_T
        else:
            E = torch.zeros((nDOF, 6), dtype=float_cp, device=DEVICE)

        b_demag = -(Fx @ m_tilde[:, 0] + Fy @ m_tilde[:, 1] + Fz @ m_tilde[:, 2])
        # Pin a DOF to 0 here instead of assembly process
        b_demag[0] = 0.0
        U = solve_cg(
            A_demag,
            b_demag,
            x0=U,
            M=Mj_demag,
            tol=tol,
            maxiter=maxiter,
            system="demag",
            use_init=use_init,
            check_every=cg_check_every,
        )

        Htilde1, Htilde2, Htilde3 = Deriv.compute_Hd_from_U(U)
        Hbar1 = -N_x * avg_m[0]
        Hbar2 = -N_y * avg_m[1]
        Hbar3 = -N_z * avg_m[2]

        # Starting the Gauss-Seidel projection algorithm
        if rot111 == False:
            m_squared = m * m
            m1_sq = m_squared[:, 0]
            m2_sq = m_squared[:, 1]
            m3_sq = m_squared[:, 2]

            f1 = (
                -2.0 * K1 * m[:, 0] * (m2_sq + m3_sq)
                + 0.5 * Hbar1
                + 0.5 * Htilde1
                + Hext1
                - 2.0 * elasC1 * m[:, 0] * (m2_sq + m3_sq)
                - 2.0 * elasC2 * (E[:, 0] - lambda100) * m[:, 0]
                - elasC3 * (E[:, 3] * m[:, 1] + E[:, 5] * m[:, 2])
            )
            f2 = (
                -2.0 * K1 * m[:, 1] * (m3_sq + m1_sq)
                + 0.5 * Hbar2
                + 0.5 * Htilde2
                + Hext2
                - 2.0 * elasC1 * m[:, 1] * (m3_sq + m1_sq)
                - 2.0 * elasC2 * (E[:, 1] - lambda100) * m[:, 1]
                - elasC3 * (E[:, 4] * m[:, 2] + E[:, 3] * m[:, 0])
            )
            f3 = (
                -2.0 * K1 * m[:, 2] * (m1_sq + m2_sq)
                + 0.5 * Hbar3
                + 0.5 * Htilde3
                + Hext3
                - 2.0 * elasC1 * m[:, 2] * (m1_sq + m2_sq)
                - 2.0 * elasC2 * (E[:, 2] - lambda100) * m[:, 2]
                - elasC3 * (E[:, 5] * m[:, 0] + E[:, 4] * m[:, 1])
            )
        else:
            m_squared = m100 * m100
            m1_sq = m_squared[:, 0]
            m2_sq = m_squared[:, 1]
            m3_sq = m_squared[:, 2]

            # First order derivatives of free energy in [100] coordinate
            v = torch.zeros((nDOF, 3), dtype=float_cp, device=DEVICE)
            v[:, 0] = (
                -2.0 * K1 * m100[:, 0] * (m2_sq + m3_sq)
                - 2.0 * elasC1 * m100[:, 0] * (m2_sq + m3_sq)
                - 2.0 * elasC2 * (E[:, 0] - lambda100) * m100[:, 0]
                - elasC3 * (E[:, 3] * m100[:, 1] + E[:, 5] * m100[:, 2])
            )
            v[:, 1] = (
                -2.0 * K1 * m100[:, 1] * (m3_sq + m1_sq)
                - 2.0 * elasC1 * m100[:, 1] * (m3_sq + m1_sq)
                - 2.0 * elasC2 * (E[:, 1] - lambda100) * m100[:, 1]
                - elasC3 * (E[:, 4] * m100[:, 2] + E[:, 3] * m100[:, 0])
            )
            v[:, 2] = (
                -2.0 * K1 * m100[:, 2] * (m1_sq + m2_sq)
                - 2.0 * elasC1 * m100[:, 2] * (m1_sq + m2_sq)
                - 2.0 * elasC2 * (E[:, 2] - lambda100) * m100[:, 2]
                - elasC3 * (E[:, 5] * m100[:, 0] + E[:, 4] * m100[:, 1])
            )
            v = v @ R.T

            f1 = v[:, 0] + 0.5 * Hbar1 + 0.5 * Htilde1 + Hext1
            f2 = v[:, 1] + 0.5 * Hbar2 + 0.5 * Htilde2 + Hext2
            f3 = v[:, 2] + 0.5 * Hbar3 + 0.5 * Htilde3 + Hext3

        f1temp = f1 * dt + m[:, 0]
        f2temp = f2 * dt + m[:, 1]
        f3temp = f3 * dt + m[:, 2]

        b_g1n = F1 @ f1temp
        b_g2n = F1 @ f2temp
        b_g3n = F1 @ f3temp

        g1n = solve_cg(
            A1,
            b_g1n,
            x0=g1n,
            M=Mj_A1,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g1n",
            use_init=use_init,
            check_every=cg_check_every,
        )
        g2n = solve_cg(
            A1,
            b_g2n,
            x0=g2n,
            M=Mj_A1,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g2n",
            use_init=use_init,
            check_every=cg_check_every,
        )
        g3n = solve_cg(
            A1,
            b_g3n,
            x0=g3n,
            M=Mj_A1,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g3n",
            use_init=use_init,
            check_every=cg_check_every,
        )

        m1star = m[:, 0] + g2n * m[:, 2] - g3n * m[:, 1]
        f1temp = f1 * dt + m1star
        b_gstar = F1 @ f1temp
        g1star = solve_cg(
            A1,
            b_gstar,
            x0=g1star,
            M=Mj_A1,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g1star",
            use_init=use_init,
            check_every=cg_check_every,
        )

        m2star = m[:, 1] + g3n * m1star - g1star * m[:, 2]
        f2temp = f2 * dt + m2star
        b_gstar = F1 @ f2temp
        g2star = solve_cg(
            A1,
            b_gstar,
            x0=g2star,
            M=Mj_A1,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g2star",
            use_init=use_init,
            check_every=cg_check_every,
        )

        m3star = m[:, 2] + g1star * m2star - g2star * m1star

        dt2 = dt * alpha_damp
        # Now moving mi_sq to be the squares of mstar
        if rot111 == False:
            m1_sq = m1star * m1star
            m2_sq = m2star * m2star
            m3_sq = m3star * m3star

            f1temp = (
                -2.0 * K1 * m1star * (m2_sq + m3_sq)
                + 0.5 * Hbar1
                + 0.5 * Htilde1
                + Hext1
                - 2.0 * elasC1 * m1star * (m2_sq + m3_sq)
                - 2.0 * elasC2 * (E[:, 0] - lambda100) * m1star
                - elasC3 * (E[:, 3] * m2star + E[:, 5] * m3star)
            ) * dt2 + m1star
            f2temp = (
                -2.0 * K1 * m2star * (m3_sq + m1_sq)
                + 0.5 * Hbar2
                + 0.5 * Htilde2
                + Hext2
                - 2.0 * elasC1 * m2star * (m3_sq + m1_sq)
                - 2.0 * elasC2 * (E[:, 1] - lambda100) * m2star
                - elasC3 * (E[:, 4] * m3star + E[:, 3] * m1star)
            ) * dt2 + m2star
            f3temp = (
                -2.0 * K1 * m3star * (m1_sq + m2_sq)
                + 0.5 * Hbar3
                + 0.5 * Htilde3
                + Hext3
                - 2.0 * elasC1 * m3star * (m1_sq + m2_sq)
                - 2.0 * elasC2 * (E[:, 2] - lambda100) * m3star
                - elasC3 * (E[:, 5] * m1star + E[:, 4] * m2star)
            ) * dt2 + m3star
        else:
            m100 = torch.stack([m1star, m2star, m3star], dim=-1) @ R

            m_squared = m100 * m100
            m1_sq = m_squared[:, 0]
            m2_sq = m_squared[:, 1]
            m3_sq = m_squared[:, 2]

            # First order derivatives of free energy in [100] coordinate
            v = torch.zeros((nDOF, 3), dtype=float_cp, device=DEVICE)
            v[:, 0] = (
                -2.0 * K1 * m100[:, 0] * (m2_sq + m3_sq)
                - 2.0 * elasC1 * m100[:, 0] * (m2_sq + m3_sq)
                - 2.0 * elasC2 * (E[:, 0] - lambda100) * m100[:, 0]
                - elasC3 * (E[:, 3] * m100[:, 1] + E[:, 5] * m100[:, 2])
            )
            v[:, 1] = (
                -2.0 * K1 * m100[:, 1] * (m3_sq + m1_sq)
                - 2.0 * elasC1 * m100[:, 1] * (m3_sq + m1_sq)
                - 2.0 * elasC2 * (E[:, 1] - lambda100) * m100[:, 1]
                - elasC3 * (E[:, 4] * m100[:, 2] + E[:, 3] * m100[:, 0])
            )
            v[:, 2] = (
                -2.0 * K1 * m100[:, 2] * (m1_sq + m2_sq)
                - 2.0 * elasC1 * m100[:, 2] * (m1_sq + m2_sq)
                - 2.0 * elasC2 * (E[:, 2] - lambda100) * m100[:, 2]
                - elasC3 * (E[:, 5] * m100[:, 0] + E[:, 4] * m100[:, 1])
            )
            v = v @ R.T

            f1temp = (v[:, 0] + 0.5 * Hbar1 + 0.5 * Htilde1 + Hext1) * dt2 + m1star
            f2temp = (v[:, 1] + 0.5 * Hbar2 + 0.5 * Htilde2 + Hext2) * dt2 + m2star
            f3temp = (v[:, 2] + 0.5 * Hbar3 + 0.5 * Htilde3 + Hext3) * dt2 + m3star

        b_gstar = F1 @ f1temp
        m1starstar = solve_cg(
            A2,
            b_gstar,
            x0=m1starstar,
            M=Mj_A2,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel m1starstar",
            use_init=use_init,
            check_every=cg_check_every,
        )

        b_gstar = F1 @ f2temp
        m2starstar = solve_cg(
            A2,
            b_gstar,
            x0=m2starstar,
            M=Mj_A2,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel m2starstar",
            use_init=use_init,
            check_every=cg_check_every,
        )

        b_gstar = F1 @ f3temp
        m3starstar = solve_cg(
            A2,
            b_gstar,
            x0=m3starstar,
            M=Mj_A2,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel m3starstar",
            use_init=use_init,
            check_every=cg_check_every,
        )

        magnitude = torch.sqrt(m1starstar**2 + m2starstar**2 + m3starstar**2)

        zero = torch.zeros((), dtype=float_cp, device=DEVICE)
        small = torch.abs(magnitude) < 1e-14
        m[:, 0] = torch.where(small, zero, m1starstar / magnitude)
        m[:, 1] = torch.where(small, zero, m2starstar / magnitude)
        m[:, 2] = torch.where(small, zero, m3starstar / magnitude)
        m[DefDOF, :] = 0.0

        sumsq = ((m - m_prev) ** 2).sum().item()
        use_init = sumsq < use_init_factor * LLG_accuracy
        m_prev = m.clone()

        if nstep % 200 == 0:
            print(f"m diff L2 norm sq: {sumsq:.10f} at LLG step {nstep}")

        avg_m = Avg.compute_average_field_gpu(m)
        m_tilde = m - avg_m

        if (
            sumsq < LLG_accuracy and nstep > llg_min_step
        ) or nstep % save_frequency == 0:
            count += 1
            Avg.write_to_paraview(
                {
                    "Htilde1": Htilde1,
                    "Htilde2": Htilde2,
                    "Htilde3": Htilde3,
                    "m1": m[:, 0],
                    "m2": m[:, 1],
                    "m3": m[:, 2],
                    "E11": E[:, 0],
                    "E22": E[:, 1],
                    "E33": E[:, 2],
                    "E12": E[:, 3],
                    "E23": E[:, 4],
                    "E13": E[:, 5],
                },
                os.path.join(output_dir, f"field_{Hext1 * ms:.0f}_{count}.vtu"),
                alpha=vtu_alpha,
            )
            hyst_file.write(
                f"{Hext1 * ms:.1f}\t{Hext2 * ms:.1f}\t{Hext3 * ms:.1f}\t"
                f"{avg_m[0].item()}\t{avg_m[1].item()}\t{avg_m[2].item()}\n"
            )
            if write_m:
                write_array(m, os.path.join(output_dir, "last_m.h5"))
            if nstep >= 5000:
                print(
                    f"LLG not converging at external field {Hext1 * ms:.0f} A/m, at LLG step {nstep}."
                )
                print(
                    f"Current average magnetization is m1 = {avg_m[0].item()}, "
                    f"m2 = {avg_m[1].item()}, m3 = {avg_m[2].item()}."
                )

        if sumsq < LLG_accuracy and nstep > llg_min_step:
            print(f"LLG converge at LLG step {nstep}.")
            print(
                f"Current average magnetization is m1 = {avg_m[0].item()}, "
                f"m2 = {avg_m[1].item()}, m3 = {avg_m[2].item()}."
            )
            current_time = time.time()
            print(
                f"Time for LLG calculation used: {current_time - LLG_start_time:.2f} s."
            )
            Hext1 -= dHext1
            Hext2 -= dHext2
            Hext3 -= dHext3
            nstep = 0
            count = 0
            if (avg_m @ stop_direction).item() > stop_value:
                print(f"Now calculating LLG with Hext1 = {Hext1 * ms:.0f} A/m.")

    hyst_file.write(
        f"{Hext1 * ms:.1f}\t{Hext2 * ms:.1f}\t{Hext3 * ms:.1f}\t"
        f"{avg_m[0].item()}\t{avg_m[1].item()}\t{avg_m[2].item()}\n"
    )
    hyst_file.close()

    print(
        f"Final state for average magnetization: m1 = {avg_m[0].item()}, "
        f"m2 = {avg_m[1].item()}, m3 = {avg_m[2].item()}."
    )
    count += 1
    Avg.write_to_paraview(
        {
            "Htilde1": Htilde1,
            "Htilde2": Htilde2,
            "Htilde3": Htilde3,
            "m1": m[:, 0],
            "m2": m[:, 1],
            "m3": m[:, 2],
            "E11": E[:, 0],
            "E22": E[:, 1],
            "E33": E[:, 2],
            "E12": E[:, 3],
            "E23": E[:, 4],
            "E13": E[:, 5],
        },
        os.path.join(output_dir, f"field_{Hext1 * ms:.0f}_{count}.vtu"),
        alpha=vtu_alpha,
    )
    if write_m:
        write_array(m, os.path.join(output_dir, "last_m.h5"))
    current_time = time.time()
    print(
        f"The code uses {current_time - micromagnetics_start_time:.2f} s in total, "
        f"{LLG_start_time - micromagnetics_start_time:.2f} s for system set up, "
        + f"and {current_time - LLG_start_time:.2f} for LLG calculation."
    )


if __name__ == "__main__":
    main()
