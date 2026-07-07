"""Numba-free, vectorized (batched-tensor) hexahedral FEM assembly prototype.

Replaces the per-element numba loops in assemble_demag / assemble_Gauss_Seidel
with batched torch einsum over all elements at once, using the exact same
tabulated shape-function data (_N_data, _dN_data) as the original so results
match to floating-point rounding.

Works on CPU or GPU; elements are processed in chunks to bound peak memory.
"""

import numpy as np
import torch


def assemble_hex_batched(
    node_coords_np,
    elements_np,
    global_id_np,
    N_data,
    dN_data,
    gauss_weights,
    device="cpu",
    chunk=524288,
    want=("K", "M", "F"),
):
    """
    Batched assembly of hexahedral element matrices in COO format.

    Parameters
    ----------
    node_coords_np : (n_nodes, 3) float64 array
    elements_np : (n_elems, 9) int array (last column = defect flag, ignored)
    global_id_np : (n_nodes,) int array (periodic DOF map)
    N_data : (8 gp, 8 nodes) shape-function table (ShapeHex._N_data)
    dN_data : (8 gp, 8 nodes, 3) derivative table (ShapeHex._dN_data)
    gauss_weights : (8,) Gauss weights
    device : torch device for the batched math
    chunk : elements per batch (bounds peak memory)
    want : which outputs to compute ("K" stiffness, "M" mass, "F" = Fx/Fy/Fz)

    Returns
    -------
    dict with keys "rows", "cols" (int32, length 64*n_elems) and, per
    request, "K", "M" (float64 values) and "F" ((3, nnz) float64 values,
    ordered x/y/z). Triplet ordering matches the original element loop
    (element-major, then a, then b).
    """
    dev = torch.device(device)
    Nt = torch.as_tensor(N_data, dtype=torch.float64, device=dev)  # (g, a)
    dNt = torch.as_tensor(dN_data, dtype=torch.float64, device=dev)  # (g, a, 3)
    w = torch.as_tensor(gauss_weights, dtype=torch.float64, device=dev)  # (g,)

    coords = torch.as_tensor(node_coords_np, dtype=torch.float64, device=dev)
    conn = torch.as_tensor(
        np.ascontiguousarray(elements_np[:, :8]), dtype=torch.long, device=dev
    )
    gid = torch.as_tensor(
        np.asarray(global_id_np), dtype=torch.long, device=dev
    )

    Ne = conn.shape[0]
    nnz = Ne * 64

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    out = {"rows": rows, "cols": cols}
    if "K" in want:
        out["K"] = np.empty(nnz, dtype=np.float64)
    if "M" in want:
        out["M"] = np.empty(nnz, dtype=np.float64)
    if "F" in want:
        out["F"] = np.empty((3, nnz), dtype=np.float64)

    eye = torch.eye(3, dtype=torch.float64, device=dev)

    for e0 in range(0, Ne, chunk):
        e1 = min(e0 + chunk, Ne)
        c = conn[e0:e1]  # (E, 8)
        C = coords[c]  # (E, 8, 3)

        # Jacobian J[e,g,d,k] = sum_a C[e,a,d] * dN[g,a,k]
        J = torch.einsum("ead,gak->egdk", C, dNt)  # (E, g, 3, 3)
        detJ = torch.linalg.det(J)  # (E, g)

        # Degenerate handling identical to the original: an element with
        # |detJ| < 1e-14 at ANY Gauss point contributes all-zero matrices.
        bad_gp = detJ.abs() < 1e-14  # (E, g)
        bad_el = bad_gp.any(dim=1)  # (E,)
        J_safe = torch.where(bad_gp[:, :, None, None], eye, J)
        Jinv = torch.linalg.inv(J_safe)  # (E, g, 3, 3)

        wdet = detJ * w  # (E, g)
        wdet = torch.where(bad_el[:, None], torch.zeros((), dtype=wdet.dtype, device=dev), wdet)

        # grad[e,g,a,d] = sum_k Jinv[e,g,d,k] * dN[g,a,k]
        grad = torch.einsum("egdk,gak->egad", Jinv, dNt)  # (E, g, 8, 3)

        sl = slice(e0 * 64, e1 * 64)
        g_ids = gid[c]  # (E, 8)
        r = g_ids[:, :, None].expand(-1, -1, 8)  # (E, 8, 8) row = a
        cc = g_ids[:, None, :].expand(-1, 8, -1)  # (E, 8, 8) col = b
        rows[sl] = r.reshape(-1).to(torch.int32).cpu().numpy()
        cols[sl] = cc.reshape(-1).to(torch.int32).cpu().numpy()

        if "K" in want:
            K_e = torch.einsum("egad,egbd,eg->eab", grad, grad, wdet)
            out["K"][sl] = K_e.reshape(-1).cpu().numpy()
        if "M" in want:
            M_e = torch.einsum("ga,gb,eg->eab", Nt, Nt, wdet)
            out["M"][sl] = M_e.reshape(-1).cpu().numpy()
        if "F" in want:
            F_e = torch.einsum("ga,egbd,eg->eabd", Nt, grad, wdet)  # (E,8,8,3)
            f = F_e.permute(3, 0, 1, 2).reshape(3, -1).cpu().numpy()
            out["F"][:, sl] = f

    return out


def impose_anchor_coo(rows, cols, vals, anchor=0):
    """Zero row/col of the anchor DOF and append a unit diagonal entry
    (same semantics as AssembleDemag.impose_anchor_node_dof0_coo)."""
    vals = vals.copy()
    vals[(rows == anchor) | (cols == anchor)] = 0.0
    rows = np.append(rows, np.int32(anchor))
    cols = np.append(cols, np.int32(anchor))
    vals = np.append(vals, 1.0)
    return rows, cols, vals
