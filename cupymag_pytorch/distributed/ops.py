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
    w = A @ r

    # One fused reduction for ||b||^2, gamma = (r,r), delta = (w,r).
    buf = torch.stack(
        [
            torch.einsum("nk,nk->k", b, b),
            torch.einsum("nk,nk->k", r, r),
            torch.einsum("nk,nk->k", w, r),
        ]
    )
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    b_norm_sq, gamma, delta = buf[0], buf[1], buf[2]

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
            p = torch.where(ok, r + beta * p, p)
            s = torch.where(ok, w + beta * s, s)  # s = A p by recurrence
            x = x + alpha * p
            r = r - alpha * s

            w = A @ r
            gamma_prev = gamma
            buf = torch.stack(
                [
                    torch.einsum("nk,nk->k", r, r),
                    torch.einsum("nk,nk->k", w, r),
                ]
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            gamma, delta = buf[0], buf[1]

            if it % check_every == 0 or it == maxiter:
                still_running = ok & (gamma > tol_sq)
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


class DistVolumeAverage(VolumeAverage):
    """Volume average over the rank-local element slice with global reduction.

    Construction uses the *global* mesh arrays (replicated in Phase 1) so
    that per-element quadrature data (corner_dofs, detJ) are bit-identical
    to the serial class; only the element rows [e0, e1) of this rank are
    kept. Field gathering uses corner indices remapped into the
    [owned | right ghost plane] local space.
    """

    def __init__(self, coords, elements, global_id, part: XSlabPartition):
        self.part = part
        elements_local = elements[part.e0 : part.e1]
        super().__init__(coords, elements_local, global_id)

        remap_r = torch.as_tensor(
            part._remap_right, dtype=torch.int64, device=self.corner_dofs.device
        )
        corner_local = remap_r[self.corner_dofs.to(torch.int64)]
        if corner_local.numel() and int(corner_local.min()) < 0:
            raise RuntimeError(
                f"Rank {part.rank}: local elements reference DOFs outside "
                "[owned + right ghost plane]."
            )
        self.corner_local = corner_local

    def compute_average_field_gpu(self, m, defect_flag=None):
        is_scalar = m.dim() == 1
        if is_scalar:
            m = m.reshape(-1, 1)

        m_ext = self.part.extend_right(m)
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
            "Use a serial VolumeAverage on rank 0 with gathered fields for "
            "VTU output (Phase 1)."
        )
