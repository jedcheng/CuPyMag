# Copyright (c) 2025-2026 Hongyi Guan
# PyTorch port of CuPyMag — distributed backend (Phase 1/2)
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

"""Distributed sparse operators, CG solver, and volume average.

``DistSparseMat`` wraps a rank-local row block split into an owned-column
part and a ghost-column part; the halo exchange is posted before the
owned-part SpMV so communication overlaps computation (world_size >= 3).

``solve_cg`` uses the Chronopoulos–Gear single-reduction CG variant: per
iteration one SpMV (``w = A r``) and ONE fused ``all_reduce`` carrying
both dot products (vs two reductions for textbook CG), with the same
multi-RHS batching, per-column breakdown freezing, and ``check_every``
host-sync throttling as the serial Phase 0 solver. Every rank sees
identical reduced scalars and takes identical branches.

``DistVolumeAverage`` evaluates the Gauss-quadrature average on the
rank-local element slice (same per-element quadrature data as the serial
class) and reduces the integral and volume globally.
"""

import torch
import torch.distributed as dist

from cupymag_pytorch.distributed.partition import XSlabPartition
from cupymag_pytorch.utils.volume_average import VolumeAverage


class DistSparseMat:
    """Row-block sparse CSR matrix, columns split into [owned | ghosts]."""

    def __init__(self, A_own, A_ghost, part: XSlabPartition):
        self.t = A_own  # (n_own, n_own); also the device/dtype reference
        self.g = A_ghost  # (n_own, 2*plane) or None (single rank)
        self.part = part

    @classmethod
    def from_scipy_global(cls, A_scipy_csr, part: XSlabPartition):
        A_own, A_ghost = part.extract_row_block(A_scipy_csr.tocsr())
        return cls(A_own, A_ghost, part)

    @property
    def shape(self):
        return self.t.shape

    def matmul_with_ghosts(self, v, ghosts):
        """SpMV given an already-exchanged (2*plane, k) ghost block."""
        y = torch.sparse.mm(self.t, v)
        if self.g is not None:
            y = y + torch.sparse.mm(self.g, ghosts)
        return y

    def matmul(self, v):
        squeeze = v.dim() == 1
        if squeeze:
            v = v.unsqueeze(1)
        # Post the halo exchange, overlap it with the owned-column SpMV.
        ctx = self.part.ghosts_start(v)
        y = torch.sparse.mm(self.t, v)
        if self.g is not None:
            y = y + torch.sparse.mm(self.g, self.part.ghosts_finish(ctx))
        return y.squeeze(1) if squeeze else y

    def __matmul__(self, v):
        return self.matmul(v)


def solve_cg(
    A,
    b,
    M=None,
    x0=None,
    tol=1e-7,
    maxiter=5000,
    use_init=False,
    system=None,
    check_every=1,
):
    """Distributed single-reduction (Chronopoulos–Gear) CG.

    Same API and semantics as the serial ``solve_cg`` (relative-residual
    tolerance per column, multi-RHS batching, breakdown freezing, final
    residual check with RuntimeError on failure). ``b`` holds this rank's
    rows; all ranks must call collectively.
    """
    b = b.to(A.t.device)
    single_rhs = b.dim() == 1
    if single_rhs:
        b = b.unsqueeze(1)

    if M is not None and M.dim() == 1:
        M = M.unsqueeze(1)  # inverse diagonal, broadcast over RHS columns

    if use_init and x0 is not None:
        x = x0.detach().clone().to(A.t.device).to(b.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        cold = False
    else:
        x = torch.zeros_like(b)
        cold = True

    zero = torch.zeros((), dtype=b.dtype, device=b.device)
    one = torch.ones((), dtype=b.dtype, device=b.device)

    r = b.clone() if cold else b - (A @ x)
    u = r if M is None else M * r
    w = A @ u

    # One fused reduction for ||b||^2, gamma = (r,u), delta = (w,u) and,
    # when preconditioned, the true residual norm rr = (r,r).
    def _fused(*pairs):
        buf = torch.stack([torch.einsum("nk,nk->k", a, c) for a, c in pairs])
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        return buf

    if M is None:
        buf = _fused((b, b), (r, u), (w, u))
        b_norm_sq, gamma, delta = buf[0], buf[1], buf[2]
        rr = gamma
    else:
        buf = _fused((b, b), (r, u), (w, u), (r, r))
        b_norm_sq, gamma, delta, rr = buf[0], buf[1], buf[2], buf[3]

    tol_sq = (tol * tol) * b_norm_sq
    nonzero = b_norm_sq > 0.0
    x = x * nonzero

    p = torch.zeros_like(r)
    s = torch.zeros_like(r)
    alpha = torch.zeros_like(gamma)
    gamma_prev = torch.ones_like(gamma)
    beta = torch.zeros_like(gamma)

    if bool(nonzero.any()):
        for it in range(1, maxiter + 1):
            if it == 1:
                # nonzero mask: zero-RHS columns stay frozen at x = 0.
                ok = (delta > 0.0) & nonzero
                alpha = torch.where(ok, gamma / torch.where(ok, delta, one), zero)
                beta = torch.zeros_like(gamma)
            else:
                pos = gamma_prev > 0.0
                beta = torch.where(
                    pos, gamma / torch.where(pos, gamma_prev, one), zero
                )
                ok = alpha > 0.0
                denom = delta - beta * gamma / torch.where(ok, alpha, one)
                ok = ok & (denom > 0.0)
                alpha = torch.where(ok, gamma / torch.where(ok, denom, one), zero)

            # Frozen columns (ok == False) keep p, s, x, r unchanged.
            p = torch.where(ok, u + beta * p, p)
            s = torch.where(ok, w + beta * s, s)  # s = A p by recurrence
            x = x + alpha * p
            r = r - alpha * s

            u = r if M is None else M * r
            w = A @ u
            gamma_prev = gamma
            if M is None:
                buf = _fused((r, u), (w, u))
                gamma, delta = buf[0], buf[1]
                rr = gamma
            else:
                buf = _fused((r, u), (w, u), (r, r))
                gamma, delta, rr = buf[0], buf[1], buf[2]

            if it % check_every == 0 or it == maxiter:
                still_running = ok & (rr > tol_sq)
                if not bool(still_running.any()):
                    break

    # Final residual check (explicit, guards recurrence drift and
    # frozen/broken-down columns).
    res = b - (A @ x)
    res_sq = torch.einsum("nk,nk->k", res, res)
    dist.all_reduce(res_sq, op=dist.ReduceOp.SUM)
    if bool((res_sq <= tol_sq).all()):
        return x.squeeze(1) if single_rhs else x

    rel = torch.sqrt(res_sq / torch.where(nonzero, b_norm_sq, one))
    worst = rel.max().item()
    if system is not None:
        msg = f"Error! CG for {system} did not converge. relative residual={worst:.3e} (tol={tol:.3e})."
    else:
        msg = f"Error! CG did not converge. relative residual={worst:.3e} (tol={tol:.3e})."
    raise RuntimeError(msg)


class DistBlockMat:
    """Distributed elasticity stiffness A_el: component-major DOFs
    [u_x; u_y; u_z], stored as a 3x3 grid of node-partitioned row blocks.

    One halo exchange of the (n_own, 3) displacement field serves all nine
    block SpMVs. The ``A @ v`` interface takes/returns the *flattened*
    component-major local vector (3*n_own,), matching the serial layout so
    ``solve_cg`` treats the coupled system as a single column.
    """

    def __init__(self, blocks, part: XSlabPartition):
        self.blocks = blocks  # blocks[da][db] = (A_own, A_ghost)
        self.part = part
        self.t = blocks[0][0][0]  # device/dtype reference

    @classmethod
    def from_scipy_global(cls, A_sp, part: XSlabPartition):
        n = part.n_dof
        A_sp = A_sp.tocsr()
        blocks = [
            [
                part.extract_rows(
                    A_sp[da * n : (da + 1) * n, db * n : (db + 1) * n].tocsr()
                )
                for db in range(3)
            ]
            for da in range(3)
        ]
        return cls(blocks, part)

    def matmul(self, v):
        squeeze = v.dim() == 1
        if not squeeze:
            v = v.squeeze(1)
        n_own = self.part.n_own
        U = v.view(3, n_own).T.contiguous()  # (n_own, 3), column = component

        ctx = self.part.ghosts_start(U)
        y = torch.empty_like(U)
        for da in range(3):
            acc = torch.sparse.mm(self.blocks[da][0][0], U[:, 0:1])
            for db in range(1, 3):
                acc = acc + torch.sparse.mm(self.blocks[da][db][0], U[:, db : db + 1])
            y[:, da] = acc.squeeze(1)
        gh = self.part.ghosts_finish(ctx)
        if gh is not None:
            for da in range(3):
                acc = torch.sparse.mm(self.blocks[da][0][1], gh[:, 0:1])
                for db in range(1, 3):
                    acc = acc + torch.sparse.mm(
                        self.blocks[da][db][1], gh[:, db : db + 1]
                    )
                y[:, da] = y[:, da] + acc.squeeze(1)

        out = y.T.reshape(-1)
        return out if squeeze else out.unsqueeze(1)

    def __matmul__(self, v):
        return self.matmul(v)


class DistFMat:
    """Distributed magnetostriction coupling F_el: three component row
    blocks over node-major stride-6 (Voigt) columns. ``matmul`` takes the
    local spontaneous strain E0 as (n_own, 6) and returns the flattened
    component-major RHS (3*n_own,), sharing one halo exchange."""

    def __init__(self, blocks, part: XSlabPartition):
        self.blocks = blocks  # blocks[da] = (F_own, F_ghost)
        self.part = part
        self.t = blocks[0][0]

    @classmethod
    def from_scipy_global(cls, F_sp, part: XSlabPartition):
        n = part.n_dof
        F_sp = F_sp.tocsr()
        blocks = [
            part.extract_rows(F_sp, row_start=da * n, col_stride=6)
            for da in range(3)
        ]
        return cls(blocks, part)

    def matmul(self, E0):
        ctx = self.part.ghosts_start(E0)
        E0f = E0.reshape(-1, 1)  # node-major, matches stride-6 column layout
        cols = [torch.sparse.mm(self.blocks[da][0], E0f) for da in range(3)]
        gh = self.part.ghosts_finish(ctx)
        if gh is not None:
            ghf = gh.reshape(-1, 1)
            cols = [
                cols[da] + torch.sparse.mm(self.blocks[da][1], ghf)
                for da in range(3)
            ]
        return torch.cat(cols, dim=0).squeeze(1)  # (3*n_own,)

    def __matmul__(self, E0):
        return self.matmul(E0)


def compute_E_from_u_dist(Fx, Fy, Fz, part, U3, R=None):
    """Voigt strains (n_own, 6) from the local displacement components
    ``U3`` (n_own, 3); mirrors ``ComputeDerivatives.compute_E_from_u``
    with a single shared halo exchange for all nine derivative SpMVs."""
    gh = part.get_ghosts(U3)

    def D(mat, c):
        g = None if gh is None else gh[:, c : c + 1]
        return mat.matmul_with_ghosts(U3[:, c : c + 1], g).squeeze(1)

    if R is None:
        E11 = D(Fx, 0)
        E22 = D(Fy, 1)
        E33 = D(Fz, 2)
        E12 = D(Fy, 0) + D(Fx, 1)
        E23 = D(Fz, 1) + D(Fy, 2)
        E13 = D(Fz, 0) + D(Fx, 2)
    else:
        dxx, dyx, dzx = D(Fx, 0), D(Fy, 0), D(Fz, 0)
        dxy, dyy, dzy = D(Fx, 1), D(Fy, 1), D(Fz, 1)
        dxz, dyz, dzz = D(Fx, 2), D(Fy, 2), D(Fz, 2)

        E11 = (R[0, 0] * dxx) + (R[1, 0] * dyx) + (R[2, 0] * dzx)
        E22 = (R[0, 1] * dxy) + (R[1, 1] * dyy) + (R[2, 1] * dzy)
        E33 = (R[0, 2] * dxz) + (R[1, 2] * dyz) + (R[2, 2] * dzz)
        E12 = (
            (R[0, 1] * dxx) + (R[1, 1] * dyx) + (R[2, 1] * dzx)
            + (R[0, 0] * dxy) + (R[1, 0] * dyy) + (R[2, 0] * dzy)
        )
        E23 = (
            (R[0, 2] * dxy) + (R[1, 2] * dyy) + (R[2, 2] * dzy)
            + (R[0, 1] * dxz) + (R[1, 1] * dyz) + (R[2, 1] * dzz)
        )
        E13 = (
            (R[0, 2] * dxx) + (R[1, 2] * dyx) + (R[2, 2] * dzx)
            + (R[0, 0] * dxz) + (R[1, 0] * dyz) + (R[2, 0] * dzz)
        )

    return torch.stack([E11, E22, E33, E12, E23, E13], dim=-1)


class DistVolumeAverage(VolumeAverage):
    """Volume average over the rank-local element slice with global reduction.

    Construction uses the *global* mesh arrays (replicated in Phase 1) so
    that per-element quadrature data (corner_dofs, detJ) are bit-identical
    to the serial class; only the element rows [e0, e1) of this rank are
    kept. Field gathering uses corner indices remapped into the
    [owned | right ghost plane] local space.
    """

    def __init__(self, coords, elements, global_id, part):
        self.part = part
        elements_local = part.select_elements(elements)
        super().__init__(coords, elements_local, global_id)

        remap = torch.as_tensor(
            part.element_remap, dtype=torch.int64, device=self.corner_dofs.device
        )
        corner_local = remap[self.corner_dofs.to(torch.int64)]
        if corner_local.numel() and int(corner_local.min()) < 0:
            raise RuntimeError(
                f"Rank {part.rank}: local elements reference DOFs outside "
                "the ghost-extended local index space."
            )
        self.corner_local = corner_local

    def compute_average_field_gpu(self, m, defect_flag=None):
        is_scalar = m.dim() == 1
        if is_scalar:
            m = m.reshape(-1, 1)

        m_ext = self.part.extend_elements(m)
        corner_U = m_ext[self.corner_local]

        MGP = torch.einsum("ebk,gb->egk", corner_U, self.N_gauss_gpu)

        volume_weights = self.detJ * self.W_gauss_gpu
        partial_integration = MGP * volume_weights[:, :, None]
        partial_integration_e = partial_integration.sum(dim=1)
        partial_volume = volume_weights.sum(dim=1)

        if defect_flag is not None:
            df_float = defect_flag.to(m.dtype)
            is_not_defect = 1.0 - df_float
            partial_integration_e = partial_integration_e * is_not_defect[:, None]
            partial_volume_summed = (partial_volume * is_not_defect).sum()
        else:
            partial_volume_summed = partial_volume.sum()

        total_integration = partial_integration_e.sum(dim=0)

        # One all_reduce for [integral components..., volume].
        buf = torch.cat([total_integration, partial_volume_summed.reshape(1)])
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        total_integration, total_volume = buf[:-1], buf[-1]

        if abs(total_volume.item()) < 1e-6:
            print("WARNING: Total volume is very small, results may be unstable")
            total_volume = max(abs(total_volume.item()), 1e-6)

        avg_vec = total_integration / total_volume
        return avg_vec[0] if is_scalar else avg_vec

    def write_to_paraview(self, *args, **kwargs):
        raise NotImplementedError(
            "Use a serial VolumeAverage on rank 0 with gathered fields, or "
            "write_to_paraview_parallel for per-rank .pvtu pieces."
        )

    def write_to_paraview_parallel(self, field_dict, filename, alpha=0.5, eps=1e-14):
        """Write this rank's element slice as a .vtu piece plus (rank 0) a
        .pvtu index referencing all pieces — no field gather needed.

        ``field_dict`` values are rank-local DOF fields (n_own,) or
        (n_own, C). Mirrors the serial ``write_to_paraview`` (including
        nodal/cell blending) restricted to the local piece; nodal/cell
        blending and periodic-DOF averaging are evaluated per piece, so
        values at rank-boundary nodes can differ slightly from a serial
        write (visualization only).
        """
        import meshio
        import numpy as np
        from scipy.sparse import coo_matrix

        from cupymag_pytorch.utils.backend import to_np

        part = self.part
        rank, P = part.rank, part.world_size
        n_per_elem = self.n_nodes_per_elem

        cells_glob = to_np(self.elements)[:, :n_per_elem].astype(np.int64)
        piece_nodes, cells = np.unique(cells_glob, return_inverse=True)
        cells = cells.reshape(cells_glob.shape)
        coords = to_np(self.coords)[piece_nodes]
        gid = to_np(self.original_global_id).astype(np.int64)[piece_nodes]

        # DOF value lookup: piece node -> ghost-extended local index.
        node_ext = np.asarray(part.element_remap)[gid]

        n_nodes, n_elems = coords.shape[0], cells.shape[0]
        N_g = to_np(self.N_gauss_gpu)
        W_g = to_np(self.W_gauss_gpu)
        detJ = to_np(self.detJ)
        cDOF = node_ext[cells]  # (e, n) ext indices per corner

        row = cells.reshape(-1)
        col = np.repeat(np.arange(n_elems), n_per_elem)
        data = np.ones_like(row, dtype=np.float64)
        A = coo_matrix((data, (row, col)), shape=(n_nodes, n_elems)).tocsr()
        A_sum = np.asarray(A.sum(axis=1)).ravel()

        uniq, inverse, counts = np.unique(gid, return_inverse=True, return_counts=True)
        dof_inv_cnt = (1.0 / counts)[inverse][:, None]

        point_data = {}
        cell_data = {}

        names = []
        for name, f in field_dict.items():
            f2 = f if f.dim() == 2 else f.reshape(-1, 1)
            f_ext = to_np(part.extend_elements(f2)).astype(np.float64)
            nC = f_ext.shape[1]

            node_val = f_ext[node_ext]  # (n_nodes, nC)

            if alpha < 0.999:
                f_e = f_ext[cDOF]  # (e, n, C)
                F_e_g = np.tensordot(f_e, N_g, axes=(1, 1)).transpose(0, 2, 1)
                weight = detJ * W_g

                num = np.einsum("eg,egc->ec", weight, F_e_g)
                den = weight.sum(axis=1, keepdims=True)

                elem_val = np.where(den > eps, num / den, f_e.mean(axis=1))

                cc_val = A @ elem_val
                cc_val = np.where(A_sum[:, None] > 0, cc_val / A_sum[:, None], node_val)

                buf = np.zeros((uniq.size, nC), dtype=cc_val.dtype)
                np.add.at(buf, inverse, cc_val)
                cc_val = buf[inverse] * dof_inv_cnt

                node_val = alpha * node_val + (1.0 - alpha) * cc_val

            if nC == 1:
                point_data[name] = node_val[:, 0]
                names.append(name)
            else:
                for c in range(nC):
                    point_data[f"{name}_{c + 1}"] = node_val[:, c]
                    names.append(f"{name}_{c + 1}")

        cell_data["defect_flag"] = [to_np(self.defect_flags).astype(np.int32)]

        base = filename[: -len(".vtu")] if filename.endswith(".vtu") else filename
        piece_file = f"{base}_p{rank}.vtu"
        elem_type = {4: "tetra", 8: "hexahedron"}[n_per_elem]
        meshio.write(
            piece_file,
            meshio.Mesh(
                points=coords,
                cells=[(elem_type, cells)],
                point_data=point_data,
                cell_data=cell_data,
            ),
        )

        if rank == 0:
            import os

            arrays = "\n".join(
                f'      <PDataArray type="Float64" Name="{n}"/>' for n in names
            )
            pieces = "\n".join(
                f'    <Piece Source="{os.path.basename(base)}_p{r}.vtu"/>'
                for r in range(P)
            )
            with open(f"{base}.pvtu", "w") as fh:
                fh.write(
                    '<?xml version="1.0"?>\n'
                    '<VTKFile type="PUnstructuredGrid" version="0.1" '
                    'byte_order="LittleEndian">\n'
                    '  <PUnstructuredGrid GhostLevel="0">\n'
                    "    <PPointData>\n"
                    f"{arrays}\n"
                    "    </PPointData>\n"
                    "    <PCellData>\n"
                    '      <PDataArray type="Int32" Name="defect_flag"/>\n'
                    "    </PCellData>\n"
                    "    <PPoints>\n"
                    '      <PDataArray type="Float64" Name="Points" '
                    'NumberOfComponents="3"/>\n'
                    "    </PPoints>\n"
                    f"{pieces}\n"
                    "  </PUnstructuredGrid>\n"
                    "</VTKFile>\n"
                )
            print(f"Field written to {base}.pvtu ({P} pieces).")
