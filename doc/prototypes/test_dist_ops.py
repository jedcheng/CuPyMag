"""Unit test for the distributed primitives (partition, halo exchange,
SpMV, CG, volume average) against replicated serial references.

Run:  mpirun -n 2 python test_dist_ops.py
      mpirun -n 4 python test_dist_ops.py
"""

import os
import sys

sys.argv = sys.argv[:1]
os.environ["CUPYMAG_CONFIG_PATH"] = (
    "/fast/pj24001684/pytorch_mpi/CuPyMag/examples/example_config_quick.yaml"
)
sys.path.insert(0, "/fast/pj24001684/pytorch_mpi/CuPyMag")

import numpy as np
import torch
import torch.distributed as dist

dist.init_process_group(backend="mpi")
rank = dist.get_rank()
P = dist.get_world_size()

from cupymag_pytorch.core.parameters import Nx, Ny, Nz, alpha, tol
from cupymag_pytorch.distributed.micromagnetics import _assemble_global_scipy
from cupymag_pytorch.distributed.ops import (
    DistSparseMat,
    DistVolumeAverage,
    solve_cg as solve_cg_dist,
)
from cupymag_pytorch.distributed.partition import XSlabPartition
from cupymag_pytorch.mesh.setup_FEM_mesh import FEMMesh
from cupymag_pytorch.solvers.linear_solvers import solve_cg as solve_cg_serial
from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.sparse_wrapper import SparseMat
from cupymag_pytorch.utils.volume_average import VolumeAverage

failures = []


def check(name, cond, detail=""):
    if rank == 0:
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


mesh = FEMMesh()
DefDOF = FEMMesh.DefDOF
nDOF = mesh.n_dof

A_demag_sp, Fx_sp, Fy_sp, Fz_sp, A1_sp, A2_sp, F1_sp = _assemble_global_scipy(
    mesh, DefDOF
)
part = XSlabPartition(Nx, Ny, Nz, mesh.global_id_np, DEVICE)

torch.manual_seed(1234)  # identical on all ranks

# ---- 1. halo exchange -------------------------------------------------
x_full = torch.arange(nDOF, dtype=torch.float64, device=DEVICE)
x_loc = x_full[part.r0 : part.r1].clone()
x_ext = part.exchange_ghosts(x_loc)
if P > 1:
    gl_expect = x_full[part.lg0 : part.lg0 + part.plane]
    gr_expect = x_full[part.rg0 : part.rg0 + part.plane]
    ok_l = torch.equal(x_ext[part.n_own : part.n_own + part.plane], gl_expect)
    ok_r = torch.equal(x_ext[part.n_own + part.plane :], gr_expect)
else:
    ok_l = ok_r = torch.equal(x_ext, x_full)
ok_t = torch.tensor([ok_l and ok_r], dtype=torch.int64)
dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
check("halo exchange planes", bool(ok_t.item()))

# ---- 2. SpMV ----------------------------------------------------------
for name, A_sp in [
    ("A_demag", A_demag_sp),
    ("A1", A1_sp),
    ("F1", F1_sp),
    ("Fx", Fx_sp),
]:
    A_dist = DistSparseMat.from_scipy_global(A_sp, part)
    v_full = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
    y_ref = torch.from_numpy(A_sp @ v_full.numpy())
    y_loc = A_dist @ v_full[part.r0 : part.r1].clone()
    y_g = part.gather_rows(y_loc)
    if rank == 0:
        err = (y_g - y_ref).abs().max().item()
    else:
        err = 0.0
    err_t = torch.tensor([err])
    dist.broadcast(err_t, src=0)
    check(f"SpMV {name}", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# multi-RHS SpMV
A1_dist = DistSparseMat.from_scipy_global(A1_sp, part)
V = torch.randn(nDOF, 3, dtype=torch.float64, device=DEVICE)
Y_ref = torch.from_numpy(A1_sp @ V.numpy())
Y_g = part.gather_rows(A1_dist @ V[part.r0 : part.r1].clone())
err = (Y_g - Y_ref).abs().max().item() if rank == 0 else 0.0
err_t = torch.tensor([err])
dist.broadcast(err_t, src=0)
check("SpMV multi-RHS", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# ---- 3. CG ------------------------------------------------------------
A1_serial = SparseMat.from_scipy(A1_sp)
b_full = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
x_serial = solve_cg_serial(A1_serial, b_full, tol=1e-10)
x_dist = solve_cg_dist(A1_dist, b_full[part.r0 : part.r1].clone(), tol=1e-10)
x_g = part.gather_rows(x_dist)
err = (x_g - x_serial).abs().max().item() if rank == 0 else 0.0
err_t = torch.tensor([err])
dist.broadcast(err_t, src=0)
check("CG A1 vs serial", err_t.item() < 1e-9, f"max|diff|={err_t.item():.2e}")

# CG on the Poisson-like demag system (anchored)
A_demag_dist = DistSparseMat.from_scipy_global(A_demag_sp, part)
A_demag_serial = SparseMat.from_scipy(A_demag_sp)
b2 = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
b2[0] = 0.0  # anchored DOF
x2_serial = solve_cg_serial(A_demag_serial, b2, tol=1e-9)
x2_dist = solve_cg_dist(A_demag_dist, b2[part.r0 : part.r1].clone(), tol=1e-9)
x2_g = part.gather_rows(x2_dist)
err = (x2_g - x2_serial).abs().max().item() if rank == 0 else 0.0
err_t = torch.tensor([err])
dist.broadcast(err_t, src=0)
check("CG A_demag vs serial", err_t.item() < 1e-7, f"max|diff|={err_t.item():.2e}")

# multi-RHS distributed CG == three single solves
B = torch.randn(nDOF, 3, dtype=torch.float64, device=DEVICE)
XB = solve_cg_dist(A1_dist, B[part.r0 : part.r1].clone(), tol=1e-10)
Xcols = torch.stack(
    [
        solve_cg_dist(A1_dist, B[part.r0 : part.r1, j].clone(), tol=1e-10)
        for j in range(3)
    ],
    dim=1,
)
err_t = torch.tensor([(XB - Xcols).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("dist CG multi-RHS", err_t.item() < 1e-8, f"max|diff|={err_t.item():.2e}")

# ---- 4. volume average -------------------------------------------------
Avg_serial = VolumeAverage(mesh.node_coords_pt, mesh.elements_pt, mesh.global_id_pt)
Avg_dist = DistVolumeAverage(
    mesh.node_coords_pt, mesh.elements_pt, mesh.global_id_pt, part
)
f_full = torch.randn(nDOF, 3, dtype=torch.float64, device=DEVICE)
a_ref = Avg_serial.compute_average_field_gpu(f_full)
a_dist = Avg_dist.compute_average_field_gpu(f_full[part.r0 : part.r1].clone())
err_t = torch.tensor([(a_dist - a_ref).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("volume average", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# scalar field variant
f1 = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
a_ref = Avg_serial.compute_average_field_gpu(f1)
a_dist = Avg_dist.compute_average_field_gpu(f1[part.r0 : part.r1].clone())
err_t = torch.tensor([abs((a_dist - a_ref).item())])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("volume average (scalar)", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# ---- 5. elasticity operators -------------------------------------------
from cupymag_pytorch.distributed.micromagnetics import _assemble_elasticity_scipy
from cupymag_pytorch.distributed.ops import (
    DistBlockMat,
    DistFMat,
    compute_E_from_u_dist,
)
from cupymag_pytorch.utils.compute_derivatives import ComputeDerivatives

A_el_sp, F_el_sp = _assemble_elasticity_scipy(mesh)
A_el = DistBlockMat.from_scipy_global(A_el_sp, part)
F_el = DistFMat.from_scipy_global(F_el_sp, part)


def slice_cm(v_full, ncomp):
    """Component-major local slice: [v_c[r0:r1] for each component]."""
    return torch.cat(
        [v_full[c * nDOF + part.r0 : c * nDOF + part.r1] for c in range(ncomp)]
    )


# A_el SpMV (relative error: normalized c11 makes entries ~1e5)
u_full = torch.randn(3 * nDOF, dtype=torch.float64, device=DEVICE)
y_ref = torch.from_numpy(A_el_sp @ u_full.numpy())
y_loc = A_el @ slice_cm(u_full, 3).clone()
scale = y_ref.abs().max().item()
err_t = torch.tensor([(y_loc - slice_cm(y_ref, 3)).abs().max().item() / scale])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("SpMV A_el (3x3 blocks)", err_t.item() < 1e-13, f"rel|diff|={err_t.item():.2e}")

# F_el SpMV (E0 node-major stride 6)
E0_full = torch.randn(nDOF, 6, dtype=torch.float64, device=DEVICE)
yF_ref = torch.from_numpy(F_el_sp @ E0_full.reshape(-1).numpy())
yF_loc = F_el @ E0_full[part.r0 : part.r1].clone()
scale = yF_ref.abs().max().item()
err_t = torch.tensor([(yF_loc - slice_cm(yF_ref, 3)).abs().max().item() / scale])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("SpMV F_el (stride-6 cols)", err_t.item() < 1e-13, f"rel|diff|={err_t.item():.2e}")

# CG on A_el (anchored: rows 0, n, 2n pinned)
b_el = torch.randn(3 * nDOF, dtype=torch.float64, device=DEVICE)
b_el[[0, nDOF, 2 * nDOF]] = 0.0
A_el_serial = SparseMat.from_scipy(A_el_sp)
x_ref = solve_cg_serial(A_el_serial, b_el, tol=1e-9)
x_loc = solve_cg_dist(A_el, slice_cm(b_el, 3).clone(), tol=1e-9)
err_t = torch.tensor([(x_loc - slice_cm(x_ref, 3)).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("CG A_el vs serial", err_t.item() < 1e-7, f"max|diff|={err_t.item():.2e}")

# compute_E_from_u vs serial (plain and rotated)
Fx_s = SparseMat.from_scipy(Fx_sp)
Fy_s = SparseMat.from_scipy(Fy_sp)
Fz_s = SparseMat.from_scipy(Fz_sp)
Deriv = ComputeDerivatives(Fx_s, Fy_s, Fz_s)
Fy_d = DistSparseMat.from_scipy_global(Fy_sp, part)
Fz_d = DistSparseMat.from_scipy_global(Fz_sp, part)
Fx_d = DistSparseMat.from_scipy_global(Fx_sp, part)
U3_loc = torch.stack(
    [u_full[c * nDOF + part.r0 : c * nDOF + part.r1] for c in range(3)], dim=1
).contiguous()
for Rm, name in [(None, "plain"), (torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0], "rotated")]:
    E_ref = Deriv.compute_E_from_u(nDOF, u_full, Rm)
    E_loc = compute_E_from_u_dist(Fx_d, Fy_d, Fz_d, part, U3_loc, Rm)
    err_t = torch.tensor([(E_loc - E_ref[part.r0 : part.r1]).abs().max().item()])
    dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
    check(f"compute_E_from_u ({name})", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# ---- summary -----------------------------------------------------------
nfail = torch.tensor([len(failures)])
dist.all_reduce(nfail, op=dist.ReduceOp.SUM)
if rank == 0:
    print()
    if nfail.item():
        print(f"FAILURES on some rank: {nfail.item()}")
    else:
        print(f"All distributed-ops tests passed with {P} ranks.")
dist.destroy_process_group()
sys.exit(1 if nfail.item() else 0)
