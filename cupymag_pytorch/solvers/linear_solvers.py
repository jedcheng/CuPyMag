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

"""Conjugate-Gradient solver implemented in PyTorch.

The original code called ``cupyx.scipy.sparse.linalg.cg``. PyTorch ships
no iterative sparse solver, so a standard CG implementation is provided
here. The operator ``A`` is expected to be a ``SparseMat`` (or anything
supporting ``A @ v``) and ``b`` a dense tensor on the same device.

The convergence criterion is the standard *relative* residual:

    ||b - A x||_2 / ||b||_2  <=  tol

Two properties matter for performance (and are prerequisites for the
distributed backend, see ``doc/distributed_approach.md``):

* All iteration scalars (alpha, beta, residual norms) stay on the device.
  The host receives a single boolean per convergence check, every
  ``check_every`` iterations, instead of two ``.item()`` round-trips per
  iteration.
* ``b`` may be 2-D ``(n, k)``: the k right-hand sides are solved
  simultaneously as *batched* CG. Each column produces the same iterates
  as a separate single-RHS solve, but kernel launches (and, in the
  distributed backend, collectives) are shared across columns.
"""

import torch


def _dots(a, b):
    """Column-wise dot products: (n, k) x (n, k) -> (k,)."""
    return torch.einsum("nk,nk->k", a, b)


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
    """
    Solve ``A x = b`` with the Conjugate-Gradient method.

    Parameters
    ----------
    A : SparseMat
        Symmetric positive-definite operator (supports ``A @ v``).
    b : torch.Tensor
        Right-hand side, 1-D ``(n,)`` or 2-D ``(n, k)`` for k simultaneous
        solves against the same operator.
    M : torch.Tensor, optional
        Jacobi preconditioner as the *inverse diagonal* of ``A``, shape
        ``(n,)`` (broadcast over RHS columns). ``None`` disables
        preconditioning. Convergence is still measured on the true
        relative residual ||b - A x|| / ||b||.
    x0 : torch.Tensor, optional
        Initial guess (same shape as ``b``); only used when ``use_init``
        is True.
    tol : float
        Relative-residual convergence tolerance (per column).
    maxiter : int
        Maximum number of CG iterations.
    use_init : bool
        Whether to start from ``x0`` instead of zero.
    system : str, optional
        Label used in error reporting.
    check_every : int
        Fetch the device-side convergence flag only every ``check_every``
        iterations. The default 1 reproduces the original convergence
        semantics; larger values reduce host synchronisation at the cost
        of up to ``check_every - 1`` extra iterations past convergence
        (which only lower the residual further).

    Returns
    -------
    torch.Tensor
        Solution with the same shape as ``b``.
    """
    b = b.to(A.t.device)
    single_rhs = b.dim() == 1
    if single_rhs:
        b = b.unsqueeze(1)

    b_norm_sq = _dots(b, b)  # (k,)
    tol_sq = (tol * tol) * b_norm_sq  # (tol * ||b||)^2 per column
    nonzero = b_norm_sq > 0.0  # zero-RHS columns stay x = 0

    if use_init and x0 is not None:
        x = x0.detach().clone().to(A.t.device).to(b.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x * nonzero
        r = b - (A @ x)
    else:
        x = torch.zeros_like(b)
        r = b.clone()

    if M is not None and M.dim() == 1:
        M = M.unsqueeze(1)  # inverse diagonal, broadcast over RHS columns

    z = r if M is None else M * r
    p = z.clone()
    rz_old = _dots(r, z)  # (k,); equals ||r||^2 when unpreconditioned
    zero = torch.zeros((), dtype=b.dtype, device=b.device)
    one = torch.ones((), dtype=b.dtype, device=b.device)

    if bool(nonzero.any()):
        for it in range(1, maxiter + 1):
            Ap = A @ p
            pAp = _dots(p, Ap)

            # ok == False marks CG breakdown (A not SPD along p) for that
            # column; the column is frozen (alpha = 0, p kept) so the final
            # residual check decides, mirroring the original early break.
            ok = pAp > 0.0
            alpha = torch.where(ok, rz_old / torch.where(ok, pAp, one), zero)
            x = x + alpha * p
            r = r - alpha * Ap

            z = r if M is None else M * r
            rz_new = _dots(r, z)
            rr = rz_new if M is None else _dots(r, r)
            pos = rz_old > 0.0
            beta = torch.where(pos, rz_new / torch.where(pos, rz_old, one), zero)
            p = torch.where(ok, z + beta * p, p)
            rz_old = rz_new

            if it % check_every == 0 or it == maxiter:
                still_running = ok & (rr > tol_sq)
                if not bool(still_running.any()):
                    break

    # Final residual check (also covers frozen / broken-down columns).
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
