"""Unit test for the distributed primitives (partition, halo exchange,
SpMV, CG, PCG, volume average, elasticity operators) against replicated
serial references. Works for both Hex (XSlabPartition) and Tet
(GeneralPartition) meshes.

Run:  mpirun -n 2 python test_dist_ops.py
      TEST_CONFIG=/path/to/tet_config.yaml mpirun -n 4 python test_dist_ops.py
"""

import os
import sys

sys.argv = sys.argv[:1]
os.environ["CUPYMAG_CONFIG_PATH"] = os.environ.get(
    "TEST_CONFIG",
    "/fast/pj24001684/pytorch_mpi/CuPyMag/examples/example_config_quick.yaml",
)
sys.path.insert(0, "/fast/pj24001684/pytorch_mpi/CuPyMag")

import numpy as np
import torch
import torch.distributed as dist

dist.init_process_group(backend="mpi")
rank = dist.get_rank()
P = dist.get_world_size()

import cupymag_pytorch.core.parameters as params
from cupymag_pytorch.distributed.micromagnetics import (
    _assemble_elasticity_scipy,
    _assemble_global_scipy,
)
from cupymag_pytorch.distributed.ops import (
    DistBlockMat,
    DistFMat,
    DistSparseMat,
    DistVolumeAverage,
    compute_E_from_u_dist,
    solve_cg as solve_cg_dist,
)
from cupymag_pytorch.distributed.partition import GeneralPartition, XSlabPartition
from cupymag_pytorch.mesh.setup_FEM_mesh import FEMMesh
from cupymag_pytorch.solvers.linear_solvers import solve_cg as solve_cg_serial
from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.compute_derivatives import ComputeDerivatives
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
is_hex = params.grid_type == "Hex"
if is_hex:
    part = XSlabPartition(params.Nx, params.Ny, params.Nz, mesh.global_id_np, DEVICE)
else:
    part = GeneralPartition(mesh, DEVICE)
if rank == 0:
    print(f"grid_type={params.grid_type}, partition={type(part).__name__}")

torch.manual_seed(1234)  # identical on all ranks


def loc(x_full):
    return part.slice_field(x_full)


# ---- 1. halo exchange (hex plane semantics only) ------------------------
if is_hex:
    x_full = torch.arange(nDOF, dtype=torch.float64, device=DEVICE)
    x_ext = part.exchange_ghosts(loc(x_full))
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
    y_loc = A_dist @ loc(v_full)
    err_t = torch.tensor([(y_loc - loc(y_ref)).abs().max().item()])
    dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
    check(f"SpMV {name}", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# multi-RHS SpMV
A1_dist = DistSparseMat.from_scipy_global(A1_sp, part)
V = torch.randn(nDOF, 3, dtype=torch.float64, device=DEVICE)
Y_ref = torch.from_numpy(A1_sp @ V.numpy())
err_t = torch.tensor([((A1_dist @ loc(V)) - loc(Y_ref)).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("SpMV multi-RHS", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# gather_rows round trip
g_full = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
g_back = part.gather_rows(loc(g_full))
ok = bool(torch.equal(g_back, g_full)) if rank == 0 else True
ok_t = torch.tensor([int(ok)])
dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
check("gather_rows round trip", bool(ok_t.item()))

# ---- 3. CG ------------------------------------------------------------
A1_serial = SparseMat.from_scipy(A1_sp)
b_full = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
x_serial = solve_cg_serial(A1_serial, b_full, tol=1e-10)
x_dist = solve_cg_dist(A1_dist, loc(b_full), tol=1e-10)
err_t = torch.tensor([(x_dist - loc(x_serial)).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("CG A1 vs serial", err_t.item() < 1e-9, f"max|diff|={err_t.item():.2e}")

# CG on the Poisson-like demag system (anchored)
A_demag_dist = DistSparseMat.from_scipy_global(A_demag_sp, part)
A_demag_serial = SparseMat.from_scipy(A_demag_sp)
b2 = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
b2[0] = 0.0  # anchored DOF
x2_serial = solve_cg_serial(A_demag_serial, b2, tol=1e-9)
x2_dist = solve_cg_dist(A_demag_dist, loc(b2), tol=1e-9)
err_t = torch.tensor([(x2_dist - loc(x2_serial)).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("CG A_demag vs serial", err_t.item() < 1e-7, f"max|diff|={err_t.item():.2e}")

# multi-RHS distributed CG == three single solves
B = torch.randn(nDOF, 3, dtype=torch.float64, device=DEVICE)
XB = solve_cg_dist(A1_dist, loc(B), tol=1e-10)
Xcols = torch.stack(
    [solve_cg_dist(A1_dist, loc(B)[:, j].clone(), tol=1e-10) for j in range(3)],
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
a_dist = Avg_dist.compute_average_field_gpu(loc(f_full))
err_t = torch.tensor([(a_dist - a_ref).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("volume average", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

f1 = torch.randn(nDOF, dtype=torch.float64, device=DEVICE)
a_ref = Avg_serial.compute_average_field_gpu(f1)
a_dist = Avg_dist.compute_average_field_gpu(loc(f1))
err_t = torch.tensor([abs((a_dist - a_ref).item())])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("volume average (scalar)", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}")

# ---- 4b. Jacobi-preconditioned CG ---------------------------------------
Mj_A1 = torch.as_tensor(
    1.0 / part.slice_rows_np(A1_sp.diagonal()), dtype=torch.float64, device=DEVICE
)
x_pcg = solve_cg_dist(A1_dist, loc(b_full), M=Mj_A1, tol=1e-10)
err_t = torch.tensor([(x_pcg - x_dist).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("dist PCG (jacobi) A1", err_t.item() < 1e-8, f"max|diff|={err_t.item():.2e}")

Mj_dm = torch.as_tensor(
    1.0 / part.slice_rows_np(A_demag_sp.diagonal()), dtype=torch.float64, device=DEVICE
)
x2_pcg = solve_cg_dist(A_demag_dist, loc(b2), M=Mj_dm, tol=1e-9)
err_t = torch.tensor([(x2_pcg - x2_dist).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("dist PCG (jacobi) A_demag", err_t.item() < 1e-6, f"max|diff|={err_t.item():.2e}")

x_pcg_s = solve_cg_serial(A1_serial, b_full, M=1.0 / A1_serial.diagonal(), tol=1e-10)
check(
    "serial PCG (jacobi) A1",
    torch.allclose(x_pcg_s, x_serial, atol=1e-8),
    f"max|diff|={(x_pcg_s - x_serial).abs().max():.2e}",
)

# ---- 5. elasticity operators -------------------------------------------
A_el_sp, F_el_sp = _assemble_elasticity_scipy(mesh)
A_el = DistBlockMat.from_scipy_global(A_el_sp, part)
F_el = DistFMat.from_scipy_global(F_el_sp, part)


def slice_cm(v_full, ncomp):
    """Component-major local slice via the partition."""
    return torch.cat(
        [part.slice_field(v_full[c * nDOF : (c + 1) * nDOF]) for c in range(ncomp)]
    )


# A_el SpMV (relative error: normalized c11 makes entries ~1e5)
u_full = torch.randn(3 * nDOF, dtype=torch.float64, device=DEVICE)
y_ref = torch.from_numpy(A_el_sp @ u_full.numpy())
y_loc = A_el @ slice_cm(u_full, 3)
scale = y_ref.abs().max().item()
err_t = torch.tensor([(y_loc - slice_cm(y_ref, 3)).abs().max().item() / scale])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("SpMV A_el (3x3 blocks)", err_t.item() < 1e-13, f"rel|diff|={err_t.item():.2e}")

# F_el SpMV (E0 node-major stride 6)
E0_full = torch.randn(nDOF, 6, dtype=torch.float64, device=DEVICE)
yF_ref = torch.from_numpy(F_el_sp @ E0_full.reshape(-1).numpy())
yF_loc = F_el @ loc(E0_full)
scale = yF_ref.abs().max().item()
err_t = torch.tensor([(yF_loc - slice_cm(yF_ref, 3)).abs().max().item() / scale])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("SpMV F_el (stride-6 cols)", err_t.item() < 1e-13, f"rel|diff|={err_t.item():.2e}")

# CG on A_el (anchored: rows 0, n, 2n pinned)
b_el = torch.randn(3 * nDOF, dtype=torch.float64, device=DEVICE)
b_el[[0, nDOF, 2 * nDOF]] = 0.0
A_el_serial = SparseMat.from_scipy(A_el_sp)
x_ref = solve_cg_serial(A_el_serial, b_el, tol=1e-9)
x_loc = solve_cg_dist(A_el, slice_cm(b_el, 3), tol=1e-9)
err_t = torch.tensor([(x_loc - slice_cm(x_ref, 3)).abs().max().item()])
dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
check("CG A_el vs serial", err_t.item() < 1e-7, f"max|diff|={err_t.item():.2e}")

# compute_E_from_u vs serial (plain and rotated)
Deriv = ComputeDerivatives(
    SparseMat.from_scipy(Fx_sp), SparseMat.from_scipy(Fy_sp), SparseMat.from_scipy(Fz_sp)
)
Fx_d = DistSparseMat.from_scipy_global(Fx_sp, part)
Fy_d = DistSparseMat.from_scipy_global(Fy_sp, part)
Fz_d = DistSparseMat.from_scipy_global(Fz_sp, part)
U3_loc = torch.stack(
    [part.slice_field(u_full[c * nDOF : (c + 1) * nDOF]) for c in range(3)], dim=1
).contiguous()
for Rm, name in [
    (None, "plain"),
    (torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0], "rotated"),
]:
    E_ref = Deriv.compute_E_from_u(nDOF, u_full, Rm)
    E_loc = compute_E_from_u_dist(Fx_d, Fy_d, Fz_d, part, U3_loc, Rm)
    err_t = torch.tensor([(E_loc - loc(E_ref)).abs().max().item()])
    dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
    check(
        f"compute_E_from_u ({name})", err_t.item() < 1e-12, f"max|diff|={err_t.item():.2e}"
    )

# ---- summary -----------------------------------------------------------
nfail = torch.tensor([len(failures)])
dist.all_reduce(nfail, op=dist.ReduceOp.SUM)
if rank == 0:
    print()
    if nfail.item():
        print(f"FAILURES on some rank: {nfail.item()}")
    else:
        print(f"All distributed-ops tests passed with {P} ranks ({params.grid_type}).")
dist.destroy_process_group()
sys.exit(1 if nfail.item() else 0)
