"""Validate + benchmark the vectorized (numba-free) hex assembly against the
existing numba-jitted assembly classes.

Usage: python test_numba_free_assembly.py [N ...]   (default: 16 32 64)
"""

import os
import sys
import time

# parameters.py inspects sys.argv for the config path; consume our own
# CLI arguments before any cupymag import.
SIZES = [int(a) for a in sys.argv[1:]] or [16, 32, 64]
sys.argv = sys.argv[:1]

os.environ["CUPYMAG_CONFIG_PATH"] = (
    "/fast/pj24001684/pytorch_mpi/CuPyMag/examples/example_config_quick.yaml"
)
sys.path.insert(0, "/fast/pj24001684/pytorch_mpi/CuPyMag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.sparse import coo_matrix

from cupymag_pytorch.mesh.gridHex import HexGrid
from cupymag_pytorch.mesh import ShapeHex
from cupymag_pytorch.physics.assemble_demag import AssembleDemag
from cupymag_pytorch.physics.assemble_Gauss_Seidel import AssembleGaussSeidel
from cupymag_pytorch.utils.numba_shim import HAS_NUMBA

from hex_assembly_torch import assemble_hex_batched, impose_anchor_coo

print(f"HAS_NUMBA = {HAS_NUMBA}")

_, gauss_w = ShapeHex.gauss_quadrature()


def to_csr(rows, cols, vals, n):
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def rel_diff(A, B):
    d = abs(A - B)
    return (d.max() if d.nnz else 0.0) / max(abs(A).max(), 1e-300)


def run_case(N, validate=True):
    print(f"\n=== mesh {N}x{N}x{N} elements ===")
    grid = HexGrid(N, N, N, 4, 4, 4)
    node_coords_pt, elements_pt = grid.std_fem_mesh()
    node_coords = node_coords_pt.cpu().numpy().astype(np.float64)
    elements = elements_pt.cpu().numpy().astype(np.int32)

    t0 = time.perf_counter()
    global_id = grid.build_periodic_node_map(node_coords)
    t_gid = time.perf_counter() - t0
    ndof = len(np.unique(global_id))
    print(f"  build_periodic_node_map (pure python): {t_gid:.2f} s, nDOF={ndof}")

    # --- reference (numba if available) ---
    demag = AssembleDemag(node_coords, elements, global_id)
    gs = AssembleGaussSeidel(node_coords, elements, global_id)

    t0 = time.perf_counter()
    rA, cA, vA = demag.build_coo_matrix_A_numba()
    t_ref_A_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    rA, cA, vA = demag.build_coo_matrix_A_numba()
    t_ref_A = time.perf_counter() - t0

    t0 = time.perf_counter()
    fx_r, fx_c, fx_v, fy_r, fy_c, fy_v, fz_r, fz_c, fz_v = (
        demag.build_coo_matrices_F_numba()
    )
    t_ref_F = time.perf_counter() - t0

    t0 = time.perf_counter()
    gr, gc, gvK, _, _, gvM = gs.build_coo_matrices_numba()
    t_ref_GS = time.perf_counter() - t0

    label = "numba" if HAS_NUMBA else "pure-python fallback"
    print(f"  reference ({label}):")
    print(f"    demag A: {t_ref_A:.2f} s (first call incl. JIT: {t_ref_A_first:.2f} s)")
    print(f"    demag Fx/Fy/Fz: {t_ref_F:.2f} s")
    print(f"    GS K+M: {t_ref_GS:.2f} s")

    # --- vectorized torch ---
    t0 = time.perf_counter()
    res = assemble_hex_batched(
        node_coords, elements, global_id,
        ShapeHex._N_data, ShapeHex._dN_data, gauss_w,
    )
    tr, tc = res["rows"], res["cols"]
    trA, tcA, tvA = impose_anchor_coo(tr, tc, res["K"])
    t_torch = time.perf_counter() - t0
    print(f"  vectorized torch (K+M+Fx/Fy/Fz+anchor, all together): {t_torch:.2f} s")
    denom = t_ref_A + t_ref_F + t_ref_GS
    print(f"  speedup vs reference (sum of ref timings): {denom / t_torch:.1f}x")

    if validate:
        A_ref = to_csr(rA, cA, vA, ndof)
        A_new = to_csr(trA, tcA, tvA, ndof)
        print(f"  A_demag rel diff:  {rel_diff(A_ref, A_new):.3e}")

        K_ref = to_csr(gr, gc, gvK, ndof)
        K_new = to_csr(tr, tc, res["K"], ndof)
        print(f"  K (GS) rel diff:   {rel_diff(K_ref, K_new):.3e}")

        M_ref = to_csr(gr, gc, gvM, ndof)
        M_new = to_csr(tr, tc, res["M"], ndof)
        print(f"  M (GS) rel diff:   {rel_diff(M_ref, M_new):.3e}")

        for i, (rr, cc2, vv, name) in enumerate(
            [(fx_r, fx_c, fx_v, "Fx"), (fy_r, fy_c, fy_v, "Fy"), (fz_r, fz_c, fz_v, "Fz")]
        ):
            F_ref = to_csr(rr, cc2, vv, ndof)
            F_new = to_csr(tr, tc, res["F"][i], ndof)
            print(f"  {name} rel diff:      {rel_diff(F_ref, F_new):.3e}")


if __name__ == "__main__":
    for N in SIZES:
        run_case(N, validate=(N <= 32))
