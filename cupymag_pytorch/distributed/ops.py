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

"""Distributed sparse operators, CG solver, and volume average.

``DistSparseMat`` wraps a rank-local row block (ghost-extended columns)
and performs halo exchange + SpMV under the same ``A @ v`` calling
convention as the serial ``SparseMat``.

``solve_cg`` mirrors ``cupymag_pytorch.solvers.linear_solvers.solve_cg``
(including multi-RHS batching, breakdown freezing and ``check_every``)
with one change: column-wise dot products are globally reduced with
``all_reduce(SUM)``, so every rank sees identical scalars and takes
identical branches.

``DistVolumeAverage`` evaluates the Gauss-quadrature average on the
rank-local element slice (using the exact same per-element quadrature
data as the serial class) and reduces the integral and volume globally.
"""

import torch
import torch.distributed as dist

from cupymag_pytorch.distributed.partition import XSlabPartition
from cupymag_pytorch.utils.volume_average import VolumeAverage


class DistSparseMat:
    """Row-block sparse CSR matrix with ghost-extended columns."""

    def __init__(self, local_csr_tensor, part: XSlabPartition):
        self.t = local_csr_tensor
        self.part = part

    @classmethod
    def from_scipy_global(cls, A_scipy_csr, part: XSlabPartition):
        return cls(part.extract_row_block(A_scipy_csr.tocsr()), part)

    @property
    def shape(self):
        return self.t.shape

    def matmul(self, v):
        v_ext = self.part.exchange_ghosts(v)
        if v_ext.dim() == 1:
            return torch.sparse.mm(self.t, v_ext.unsqueeze(1)).squeeze(1)
        return torch.sparse.mm(self.t, v_ext)

    def __matmul__(self, v):
        return self.matmul(v)


def _dots(a, b):
    """Column-wise dot products reduced over all ranks: (n,k)x(n,k)->(k,)."""
    d = torch.einsum("nk,nk->k", a, b)
    dist.all_reduce(d, op=dist.ReduceOp.SUM)
    return d


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
    """Distributed CG on row-partitioned ``A`` (see serial ``solve_cg``).

    ``b`` holds this rank's rows, 1-D ``(n_own,)`` or 2-D ``(n_own, k)``.
    All ranks must call collectively; every rank returns its row block of
    the solution.
    """
    b = b.to(A.t.device)
    single_rhs = b.dim() == 1
    if single_rhs:
        b = b.unsqueeze(1)

    b_norm_sq = _dots(b, b)  # (k,) identical on all ranks
    tol_sq = (tol * tol) * b_norm_sq
    nonzero = b_norm_sq > 0.0

    if use_init and x0 is not None:
        x = x0.detach().clone().to(A.t.device).to(b.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x * nonzero
        r = b - (A @ x)
    else:
        x = torch.zeros_like(b)
        r = b.clone()

    p = r.clone()
    rs_old = _dots(r, r)
    zero = torch.zeros((), dtype=b.dtype, device=b.device)
    one = torch.ones((), dtype=b.dtype, device=b.device)

    if bool(nonzero.any()):
        for it in range(1, maxiter + 1):
            Ap = A @ p
            pAp = _dots(p, Ap)

            ok = pAp > 0.0
            alpha = torch.where(ok, rs_old / torch.where(ok, pAp, one), zero)
            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = _dots(r, r)
            pos = rs_old > 0.0
            beta = torch.where(pos, rs_new / torch.where(pos, rs_old, one), zero)
            p = torch.where(ok, r + beta * p, p)
            rs_old = rs_new

            if it % check_every == 0 or it == maxiter:
                still_running = ok & (rs_new > tol_sq)
                if not bool(still_running.any()):
                    break

    res = b - (A @ x)
    res_sq = _dots(res, res)
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
