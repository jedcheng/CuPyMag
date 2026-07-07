# Copyright (c) 2025-2026 Hongyi Guan
# PyTorch port of CuPyMag — distributed backend (Phase 1)
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

"""Distributed LLG main loop (Gauss-Seidel projection method).

Mirrors ``cupymag_pytorch.core.Micromagnetics.main`` step for step. The
FEM system is assembled replicated on the CPU (scipy CSR); each rank then
keeps only its x-slab row block (``XSlabPartition``). All per-DOF algebra
is rank-local; CG dot products, the LLG convergence norm and volume
averages are globally reduced, so every rank sees identical scalars and
takes identical control-flow branches. Output (VTU / HDF5 / hysteresis)
is written by rank 0 from gathered fields.

Phase 1 scope: Hex mesh, magnetoelastic coupling (ME) not supported.
"""

import os
import time

import numpy as np
import torch
import torch.distributed as dist
from scipy.sparse import coo_matrix

from cupymag_pytorch.core.parameters import *
from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.precision_select import get_float_type

float_cp = get_float_type(precision, backend="torch")

from cupymag_pytorch.distributed.ops import (
    DistSparseMat,
    DistVolumeAverage,
    solve_cg,
)
from cupymag_pytorch.distributed.partition import XSlabPartition
from cupymag_pytorch.mesh.setup_FEM_mesh import FEMMesh
from cupymag_pytorch.physics.assemble_demag import AssembleDemag
from cupymag_pytorch.physics.assemble_Gauss_Seidel import AssembleGaussSeidel
from cupymag_pytorch.utils.magnetization_io import initialize_m, write_array
from cupymag_pytorch.utils.print_logo import print_logo
from cupymag_pytorch.utils.print_system_info import print_system_info_summary
from cupymag_pytorch.utils.rot_111_matrices import get_M_matrix, get_R_matrix
from cupymag_pytorch.utils.sparse_wrapper import (
    enforce_defect_region_A_scipy,
    enforce_defect_region_F_scipy,
)
from cupymag_pytorch.utils.volume_average import VolumeAverage


def _csr(rows, cols, vals, n):
    return coo_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(n, n),
    ).tocsr()


def _assemble_global_scipy(mesh, defect_dofs):
    """Replicated CPU assembly of all global operators as scipy CSR
    (mirrors SimulationOperators + final_assembly, stopping before the
    device upload)."""
    n = mesh.n_dof
    alpha1 = alpha
    alpha2 = alpha * alpha_damp
    dd = defect_dofs.detach().cpu().numpy().astype(np.int64)

    demag = AssembleDemag(mesh.node_coords_np, mesh.elements_np, mesh.global_id_np)
    r, c, v = demag.build_coo_matrix_A_numba()
    A_demag = _csr(r, c, v, n)

    fx_r, fx_c, fx_v, fy_r, fy_c, fy_v, fz_r, fz_c, fz_v = (
        demag.build_coo_matrices_F_numba()
    )
    Fx = _csr(fx_r, fx_c, fx_v, n)
    Fy = _csr(fy_r, fy_c, fy_v, n)
    Fz = _csr(fz_r, fz_c, fz_v, n)

    gs = AssembleGaussSeidel(mesh.node_coords_np, mesh.elements_np, mesh.global_id_np)
    gr, gc, gvK, fr, fc, gvF = gs.build_coo_matrices_numba()
    A1 = enforce_defect_region_A_scipy(_csr(gr, gc, gvK * alpha1 + gvF, n), dd)
    A2 = enforce_defect_region_A_scipy(_csr(gr, gc, gvK * alpha2 + gvF, n), dd)
    F1 = enforce_defect_region_F_scipy(_csr(fr, fc, gvF, n), dd)

    return A_demag, Fx, Fy, Fz, A1, A2, F1


def main():
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized; launch via "
            "python -m cupymag_pytorch.distributed <config.yaml>"
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    root = rank == 0

    if ME:
        raise NotImplementedError(
            "Magnetoelastic coupling is not supported by the distributed "
            "backend yet (Phase 3); set magnetoelastic_coupling: false."
        )

    if root:
        print_logo()
        print_system_info_summary(config_path)
        print(f"[distributed] world_size={world_size}, backend={dist.get_backend()}")

    global Hext1, Hext2, Hext3

    if root and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(
            f"Directory {output_dir} did not exist previously, create directory {output_dir}."
        )
    dist.barrier()

    micromagnetics_start_time = time.time()

    # Rotation matrix [100] -> [111]: y = R x, x in [100], y in [111]
    if rot111:
        R = get_R_matrix(backend="torch", dtype=float_cp)

    # Replicated mesh + assembly, then per-rank row blocks.
    mesh = FEMMesh()
    DefDOF = FEMMesh.DefDOF
    nDOF = mesh.n_dof
    if root:
        print(
            f"Initialized {grid_type} FEM mesh with #nodes={mesh.n_nodes}, "
            f"#elements={mesh.n_elements}, "
            f"#defect elements={int(torch.sum(mesh.elements_pt[:, -1]).item())}."
        )

    A_demag_sp, Fx_sp, Fy_sp, Fz_sp, A1_sp, A2_sp, F1_sp = _assemble_global_scipy(
        mesh, DefDOF
    )

    part = XSlabPartition(Nx, Ny, Nz, mesh.global_id_np, DEVICE)
    if root:
        print(
            f"[distributed] x-slab partition: {part.Nx} planes over "
            f"{world_size} ranks, plane size {part.plane} DOFs."
        )

    A_demag = DistSparseMat.from_scipy_global(A_demag_sp, part)
    Fx = DistSparseMat.from_scipy_global(Fx_sp, part)
    Fy = DistSparseMat.from_scipy_global(Fy_sp, part)
    Fz = DistSparseMat.from_scipy_global(Fz_sp, part)
    A1 = DistSparseMat.from_scipy_global(A1_sp, part)
    A2 = DistSparseMat.from_scipy_global(A2_sp, part)
    F1 = DistSparseMat.from_scipy_global(F1_sp, part)
    del A_demag_sp, Fx_sp, Fy_sp, Fz_sp, A1_sp, A2_sp, F1_sp

    Avg = DistVolumeAverage(
        mesh.node_coords_pt, mesh.elements_pt, mesh.global_id_pt, part
    )
    # Serial averaging class on rank 0 for VTU output of gathered fields.
    AvgOut = (
        VolumeAverage(mesh.node_coords_pt, mesh.elements_pt, mesh.global_id_pt)
        if root
        else None
    )

    # Initialize magnetization (replicated init, then slice own rows).
    m_full = initialize_m(
        restart_m, restart, initial_magnetization, nDOF, DefDOF, float_cp
    )
    m = m_full[part.r0 : part.r1].clone()
    del m_full
    DefDOF_local = part.local_defect_dofs(DefDOF)

    avg_m = Avg.compute_average_field_gpu(m)
    m_tilde = m - avg_m

    # CG solver parameter
    use_init = False

    # Gauss-Seidel step counters
    nstep = 0
    count = 0

    llg_min_step = 100
    LLG_accuracy = nDOF * LLG_accuracy_factor

    m_prev = m.clone()

    U = None
    Gn = None  # batched g1n/g2n/g3n warm start (n_own, 3)
    g1star = None
    g2star = None
    Mss = None  # batched m*starstar warm start (n_own, 3)

    hyst_file = None
    if root:
        hyst_file = open(os.path.join(output_dir, "hysteresis.txt"), "w", buffering=1)
        hyst_file.write("Hext1\tHext2\tHext3\tavg_m1\tavg_m2\tavg_m3\n")

    def write_outputs(Htilde1, Htilde2, Htilde3, E, count):
        """Gather fields and write VTU / h5 on rank 0 (collective)."""
        m_g = part.gather_rows(m)
        Ht1_g = part.gather_rows(Htilde1)
        Ht2_g = part.gather_rows(Htilde2)
        Ht3_g = part.gather_rows(Htilde3)
        E_g = part.gather_rows(E)
        if root:
            AvgOut.write_to_paraview(
                {
                    "Htilde1": Ht1_g,
                    "Htilde2": Ht2_g,
                    "Htilde3": Ht3_g,
                    "m1": m_g[:, 0],
                    "m2": m_g[:, 1],
                    "m3": m_g[:, 2],
                    "E11": E_g[:, 0],
                    "E22": E_g[:, 1],
                    "E33": E_g[:, 2],
                    "E12": E_g[:, 3],
                    "E23": E_g[:, 4],
                    "E13": E_g[:, 5],
                },
                os.path.join(output_dir, f"field_{Hext1 * ms:.0f}_{count}.vtu"),
                alpha=vtu_alpha,
            )
            if write_m:
                write_array(m_g, os.path.join(output_dir, "last_m.h5"))

    LLG_start_time = time.time()

    if root:
        print(
            f"FEM system set up done using {LLG_start_time - micromagnetics_start_time:.2f} s, "
            f"begin LLG iterations."
        )
    while (avg_m @ stop_direction).item() > stop_value:
        nstep += 1
        if rot111 == True:
            m100 = m @ R

        # ME is not supported: strains are zero on the local rows.
        E = torch.zeros((part.n_own, 6), dtype=float_cp, device=DEVICE)

        # One shared halo exchange for the three F SpMVs on m_tilde.
        gh = part.get_ghosts(m_tilde)
        b_demag = -(
            Fx.matmul_with_ghosts(m_tilde[:, 0:1], None if gh is None else gh[:, 0:1])
            + Fy.matmul_with_ghosts(m_tilde[:, 1:2], None if gh is None else gh[:, 1:2])
            + Fz.matmul_with_ghosts(m_tilde[:, 2:3], None if gh is None else gh[:, 2:3])
        ).squeeze(1)
        # Pin a DOF to 0 here instead of assembly process
        if part.owns_dof0:
            b_demag[0] = 0.0
        U = solve_cg(
            A_demag,
            b_demag,
            x0=U,
            tol=tol,
            maxiter=maxiter,
            system="demag",
            use_init=use_init,
        )

        # One shared halo exchange for the three derivative SpMVs on U.
        U2 = U.unsqueeze(1)
        ghU = part.get_ghosts(U2)
        Htilde1 = -Fx.matmul_with_ghosts(U2, ghU).squeeze(1)
        Htilde2 = -Fy.matmul_with_ghosts(U2, ghU).squeeze(1)
        Htilde3 = -Fz.matmul_with_ghosts(U2, ghU).squeeze(1)
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
            )
            f2 = (
                -2.0 * K1 * m[:, 1] * (m3_sq + m1_sq)
                + 0.5 * Hbar2
                + 0.5 * Htilde2
                + Hext2
            )
            f3 = (
                -2.0 * K1 * m[:, 2] * (m1_sq + m2_sq)
                + 0.5 * Hbar3
                + 0.5 * Htilde3
                + Hext3
            )
        else:
            m_squared = m100 * m100
            m1_sq = m_squared[:, 0]
            m2_sq = m_squared[:, 1]
            m3_sq = m_squared[:, 2]

            # First order derivatives of free energy in [100] coordinate
            v = torch.zeros((part.n_own, 3), dtype=float_cp, device=DEVICE)
            v[:, 0] = -2.0 * K1 * m100[:, 0] * (m2_sq + m3_sq)
            v[:, 1] = -2.0 * K1 * m100[:, 1] * (m3_sq + m1_sq)
            v[:, 2] = -2.0 * K1 * m100[:, 2] * (m1_sq + m2_sq)
            v = v @ R.T

            f1 = v[:, 0] + 0.5 * Hbar1 + 0.5 * Htilde1 + Hext1
            f2 = v[:, 1] + 0.5 * Hbar2 + 0.5 * Htilde2 + Hext2
            f3 = v[:, 2] + 0.5 * Hbar3 + 0.5 * Htilde3 + Hext3

        # Batched 3-RHS solve: g1n/g2n/g3n share A1 (one exchange for the
        # RHS SpMV, and each CG iteration's collectives amortize over the
        # three columns).
        Ftemp = torch.stack([f1 * dt + m[:, 0], f2 * dt + m[:, 1], f3 * dt + m[:, 2]], dim=1)
        B_gn = F1 @ Ftemp

        Gn = solve_cg(
            A1,
            B_gn,
            x0=Gn,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g1n/g2n/g3n",
            use_init=use_init,
        )
        g1n, g2n, g3n = Gn[:, 0], Gn[:, 1], Gn[:, 2]

        m1star = m[:, 0] + g2n * m[:, 2] - g3n * m[:, 1]
        f1temp = f1 * dt + m1star
        b_gstar = F1 @ f1temp
        g1star = solve_cg(
            A1,
            b_gstar,
            x0=g1star,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g1star",
            use_init=use_init,
        )

        m2star = m[:, 1] + g3n * m1star - g1star * m[:, 2]
        f2temp = f2 * dt + m2star
        b_gstar = F1 @ f2temp
        g2star = solve_cg(
            A1,
            b_gstar,
            x0=g2star,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel g2star",
            use_init=use_init,
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
            ) * dt2 + m1star
            f2temp = (
                -2.0 * K1 * m2star * (m3_sq + m1_sq)
                + 0.5 * Hbar2
                + 0.5 * Htilde2
                + Hext2
            ) * dt2 + m2star
            f3temp = (
                -2.0 * K1 * m3star * (m1_sq + m2_sq)
                + 0.5 * Hbar3
                + 0.5 * Htilde3
                + Hext3
            ) * dt2 + m3star
        else:
            m100 = torch.stack([m1star, m2star, m3star], dim=-1) @ R

            m_squared = m100 * m100
            m1_sq = m_squared[:, 0]
            m2_sq = m_squared[:, 1]
            m3_sq = m_squared[:, 2]

            v = torch.zeros((part.n_own, 3), dtype=float_cp, device=DEVICE)
            v[:, 0] = -2.0 * K1 * m100[:, 0] * (m2_sq + m3_sq)
            v[:, 1] = -2.0 * K1 * m100[:, 1] * (m3_sq + m1_sq)
            v[:, 2] = -2.0 * K1 * m100[:, 2] * (m1_sq + m2_sq)
            v = v @ R.T

            f1temp = (v[:, 0] + 0.5 * Hbar1 + 0.5 * Htilde1 + Hext1) * dt2 + m1star
            f2temp = (v[:, 1] + 0.5 * Hbar2 + 0.5 * Htilde2 + Hext2) * dt2 + m2star
            f3temp = (v[:, 2] + 0.5 * Hbar3 + 0.5 * Htilde3 + Hext3) * dt2 + m3star

        # Batched 3-RHS solve: the m*starstar systems share A2.
        Ftemp = torch.stack([f1temp, f2temp, f3temp], dim=1)
        B_ss = F1 @ Ftemp
        Mss = solve_cg(
            A2,
            B_ss,
            x0=Mss,
            tol=tol,
            maxiter=maxiter,
            system="Gauss-Seidel m*starstar",
            use_init=use_init,
        )
        m1starstar, m2starstar, m3starstar = Mss[:, 0], Mss[:, 1], Mss[:, 2]

        magnitude = torch.sqrt(m1starstar**2 + m2starstar**2 + m3starstar**2)

        zero = torch.zeros((), dtype=float_cp, device=DEVICE)
        small = torch.abs(magnitude) < 1e-14
        m[:, 0] = torch.where(small, zero, m1starstar / magnitude)
        m[:, 1] = torch.where(small, zero, m2starstar / magnitude)
        m[:, 2] = torch.where(small, zero, m3starstar / magnitude)
        m[DefDOF_local, :] = 0.0

        sumsq_t = ((m - m_prev) ** 2).sum()
        dist.all_reduce(sumsq_t, op=dist.ReduceOp.SUM)
        sumsq = sumsq_t.item()
        use_init = sumsq < use_init_factor * LLG_accuracy
        m_prev = m.clone()

        if root and nstep % 200 == 0:
            print(f"m diff L2 norm sq: {sumsq:.10f} at LLG step {nstep}")

        avg_m = Avg.compute_average_field_gpu(m)
        m_tilde = m - avg_m

        if (
            sumsq < LLG_accuracy and nstep > llg_min_step
        ) or nstep % save_frequency == 0:
            count += 1
            write_outputs(Htilde1, Htilde2, Htilde3, E, count)
            if root:
                hyst_file.write(
                    f"{Hext1 * ms:.1f}\t{Hext2 * ms:.1f}\t{Hext3 * ms:.1f}\t"
                    f"{avg_m[0].item()}\t{avg_m[1].item()}\t{avg_m[2].item()}\n"
                )
            if root and nstep >= 5000:
                print(
                    f"LLG not converging at external field {Hext1 * ms:.0f} A/m, at LLG step {nstep}."
                )
                print(
                    f"Current average magnetization is m1 = {avg_m[0].item()}, "
                    f"m2 = {avg_m[1].item()}, m3 = {avg_m[2].item()}."
                )

        if sumsq < LLG_accuracy and nstep > llg_min_step:
            if root:
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
            if root and (avg_m @ stop_direction).item() > stop_value:
                print(f"Now calculating LLG with Hext1 = {Hext1 * ms:.0f} A/m.")

    if root:
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
    E = torch.zeros((part.n_own, 6), dtype=float_cp, device=DEVICE)
    write_outputs(Htilde1, Htilde2, Htilde3, E, count)
    if root:
        current_time = time.time()
        print(
            f"The code uses {current_time - micromagnetics_start_time:.2f} s in total, "
            f"{LLG_start_time - micromagnetics_start_time:.2f} s for system set up, "
            + f"and {current_time - LLG_start_time:.2f} for LLG calculation."
        )
