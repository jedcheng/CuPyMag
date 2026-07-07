# Distributing CuPyMag-PyTorch — Analysis and Plan

This document analyses how to convert the single-device PyTorch port of CuPyMag
(`cupymag_pytorch/`) into a multi-device (multi-GPU / multi-node) simulator using
`torch.distributed`, building on the experience and infrastructure of
**magnum.np.distributed** (x-slab decomposition, halo exchange, all-reduce-synchronised
control flow; see `magnum-np-distributed/docs/distributed_internals.md` and the
accompanying paper).

---

## 1. Why the approach differs from magnum.np.distributed

magnum.np is a **finite-difference** code whose dominant cost is the FFT-convolution
demagnetisation field. Its distribution strategy was therefore
"halo exchange for stencil fields + slab-decomposed FFT with two all-to-all
collectives per demag evaluation".

CuPyMag is a **finite-element** code with *no FFT anywhere*. The demag field is
obtained from a scalar-potential Poisson problem solved with Conjugate Gradient
(CG), and the Gauss–Seidel projection method (GSPM) turns every LLG step into a
sequence of sparse linear solves. Per LLG step (`cupymag_pytorch/core/Micromagnetics.py`):

| # | Operation | System | Size |
|---|---|---|---|
| 1 | Demag potential | CG on `A_demag` (Poisson) | n × n |
| 2 | GSPM `g1n, g2n, g3n` | 3 CG solves on `A1` | n × n each |
| 3 | GSPM `g1star, g2star` | 2 CG solves on `A1` | n × n each |
| 4 | GSPM `m*starstar` | 3 CG solves on `A2` | n × n each |
| 5 | Elasticity (if `ME`) | CG on `A_el` | 3n × 3n |
| 6 | RHS assembly | SpMVs with `F1`, `Fx/Fy/Fz`, `F_el` | — |
| 7 | Field/strain recovery | SpMVs with `Fx/Fy/Fz` (`ComputeDerivatives`) | — |
| 8 | Volume averages | element-wise partial sums + global sum | — |
| 9 | LLG algebra, normalisation, convergence check | element-wise + global reductions | — |

All matrices are assembled **once** on the CPU (Numba, COO → scipy CSR) and then
live on the device as `torch.sparse_csr` tensors (`utils/sparse_wrapper.py`).

**Consequence:** the single primitive that must be distributed is
*CG over a row-partitioned sparse CSR matrix*. Everything else is either
embarrassingly parallel (element-wise LLG algebra) or a global reduction —
patterns that transfer directly from magnum.np.distributed.

---

## 2. Decomposition design: row-block DOF partition (x-slabs on Hex)

### 2.1 The Hex mesh gives contiguous slabs for free

`mesh/gridHex.py` numbers nodes lexicographically with the x-index **slowest**:

```
node_id = i * (Ny+1) * (Nz+1) + j * (Nz+1) + k
```

Therefore:

1. **An x-slab of node planes is a contiguous range of DOF indices** — i.e. a
   contiguous block of rows in every assembled CSR operator. The
   `DistributedMesh._compute_x_partition` logic from magnum.np.distributed
   carries over almost verbatim, applied to the `Nx+1` node planes.
2. **Trilinear hex elements couple nodes only within ±1 x-plane**, so the
   off-diagonal coupling of each rank's row block touches exactly one ghost
   plane — `(Ny+1) × (Nz+1)` values — per side. This is the same 1-wide halo
   pattern as `DistributedMesh.exchange_halos` (batched non-blocking
   `isend`/`irecv`), reusable as-is at the node-plane level.

### 2.2 Distributed SpMV

Each rank stores its row block split into two pieces:

- `A_local` — columns the rank owns,
- `A_halo` — columns living in the ghost planes of the two neighbours.

One distributed SpMV = launch halo exchange of ghost planes → compute
`A_local @ x_own` while communication is in flight → add `A_halo @ x_ghost`.

### 2.3 Distributed CG

On top of the distributed SpMV, CG needs exactly one more ingredient: the two
dot products per iteration (`pᵀAp`, `rᵀr`) become *local dot + `all_reduce(SUM)`*
— the identical pattern to the `DistributedMinimizerBB` reductions in
magnum.np.distributed. Every rank sees the same reduced scalars, so all ranks
take identical branches (convergence, breakdown, early exit): the established
"all ranks make the same decision" principle from the distributed RKF45 error
control.

### 2.4 Mapping of the remaining operations

| CuPyMag operation | Distributed treatment | magnum.np.distributed analogue |
|---|---|---|
| RHS SpMVs (`F1@f`, `Fx@m_tilde`, …) | same halo'd SpMV | halo exchange |
| `Avg.compute_average_field_gpu` | per-rank partial sums over owned elements + `all_reduce(SUM)` | energy reduction in `DistributedDemagField.E()` |
| `sumsq` convergence + stop condition | `all_reduce(SUM)`, identical control flow on all ranks | RKF45 `all_reduce(MAX)` |
| `b_demag[0] = 0` DOF pinning | applied only by the rank owning global DOF 0 | rank-aware origin fix in the slab kernel |
| VTU / hysteresis / `last_m.h5` output | gather to rank 0 (Phase 1); per-rank `.pvtu` pieces later | `DistributedScalarLogger` |
| One-time assembly | Phase 1: every rank assembles the full scipy CSR on CPU, slices its own row block + halo index maps, ships to GPU | replicated → distributed kernel evolution |

Elements (for volume averages) are assigned to the rank owning their lower
x-plane; boundary elements read corner values from the already-exchanged halo
plane.

---

## 3. The performance problem to design around: collective latency

The magnum.np.distributed paper found that *halo-bound* fields (exchange, DMI)
only gained meaningful speedup beyond ~8M cells because of kernel-dispatch and
P2P latency. CuPyMag's communication profile is **worse in that dimension**:
a naive port performs, per LLG step,

```
~9 CG solves × (tens–hundreds of iterations) × (1 halo exchange + 2 all-reduces + 1 host sync)
```

i.e. thousands of latency-bound collectives per LLG step. Mitigations, in order
of value:

1. **Kill the per-iteration host syncs in the CG loop** (`solvers/linear_solvers.py`
   used `.item()` on `pAp` and `rsnew` every iteration). Keep α, β, residual
   norms as device tensors; transfer a single boolean per convergence check,
   and allow checking only every *k*-th iteration. This already matters on a
   single GPU and is fatal in distributed mode. → **Phase 0, done.**
2. **Fuse reductions:** stack `pᵀAp` and `rᵀr` into one tensor per iteration →
   a single `all_reduce`; or adopt a single-reduction / pipelined CG variant
   (Chronopoulos–Gear) which also overlaps the reduction with the SpMV.
3. **Batch same-matrix solves as multi-RHS CG:** `g1n/g2n/g3n` share `A1`,
   the three `m*starstar` solves share `A2`. Solving 3 RHS together turns the
   SpMV into `A @ (n×3)` and amortises every collective 3×, roughly halving
   the per-step collective count. → **solver support added in Phase 0.**
4. **Overlap halo exchange with the local-block SpMV**; eventually capture the
   CG body in a CUDA graph (NCCL-only scope makes this viable).

**Sizing check:** for a 512³-node Hex mesh, one halo plane is ~263k doubles
≈ 2.1 MB per side per SpMV — bandwidth is a non-issue on NVLink; latency and
dispatch dominate, exactly like the exchange-field scaling in the paper. Expect
the same qualitative curve: scaling pays off at large meshes — which is also
where the real motivation lives: **memory capacity**. Row-partitioning drops
per-rank storage of all seven-plus operators (and the 3n×3n elasticity matrix,
the largest) by ~1/P, unlocking problem sizes a single GPU cannot hold.

---

## 4. What does not carry over / needs new work

- **Tet meshes** are unstructured, so contiguous slabs do not come for free.
  Standard route: METIS partitioning + node renumbering so each rank owns a
  contiguous ID range; the halo is the set of off-rank column indices in the
  local rows. Deferred until the Hex path validates the solver infrastructure.
- **Elasticity system layout:** `u` is stored as concatenated component blocks
  `[u_x; u_y; u_z]` (3n) and `F_el` maps 6n → 3n. With block layout each rank
  owns three non-contiguous slices; either do a batched 3-component halo
  exchange, or renumber to node-major interleaved layout at assembly time so
  rows stay contiguous (cleaner, but touches the Numba assembly).
- **No preconditioner exists** in `solve_cg`. A per-rank block-Jacobi (or
  IC(0)-on-own-block) preconditioner needs zero extra communication and cuts
  iteration counts, directly multiplying down the collective count — the
  highest-leverage optimisation after Phase 1.
- **Module-level config state** (`from parameters import *`) is harmless for
  distribution (every rank loads the same YAML), but printing/logging needs a
  rank-aware wrapper (cf. `logging_helpers` in magnum.np.distributed).

---

## 5. Phased plan

| Phase | Scope | Status |
|---|---|---|
| **0** | Serial prep: de-sync the CG loop (device-side scalars, periodic single-boolean convergence checks), add multi-RHS (batched) support to `solve_cg`. Improves single-GPU performance; prerequisite for distribution. | **done** |
| **1** | Hex mesh, correctness: replicated CPU assembly → per-rank row-block extraction + halo maps → distributed SpMV/CG/reductions → rank-0 I/O. Validate against serial with tolerance-vs-world-size methodology (cf. `sp4_with_comparison.py`). | **done** |
| **2** | Performance: fused/single-reduction CG, comm–compute overlap, use multi-RHS CG for the 3-RHS groups in `Micromagnetics.py`, benchmark with the synthetic-scaling protocol from the paper. | **done** (GPU/multi-node benchmark pending) |
| **3** | Generality: Tet/METIS path, distributed elasticity, block-Jacobi preconditioning, parallel VTU output. | planned |

### Phase 1 details (implemented)

New package `cupymag_pytorch/distributed/`, launched as::

    mpirun -n 4 python -m cupymag_pytorch.distributed config.yaml --backend=mpi
    torchrun --nproc_per_node=4 -m cupymag_pytorch.distributed config.yaml --backend=nccl

- **`partition.py` — `XSlabPartition`.** Exploits two verified properties of
  the Hex mesh: the periodic DOF numbering is lexicographic with the x-plane
  slowest (`dof(i,j,k) = i·Ny·Nz + j·Nz + k`, asserted at startup), and
  elements are emitted x-column-major. Each rank owns a contiguous run of
  x-planes → a contiguous block of matrix rows *and* elements. The mesh is
  **periodic in x** (`build_periodic_node_map` wraps the max faces), so the
  rank topology is a ring: every rank exchanges one ghost plane (`Ny·Nz`
  DOFs) with each neighbour, including rank 0 ↔ rank P−1. The halo exchange
  runs in **two batched phases** (rightward flow, then leftward flow) so each
  directed rank pair carries exactly one message per phase — unambiguous even
  at P = 2, where both neighbours are the same rank (a single-phase design
  would cross-match the messages). Each rank must own ≥ 2 planes (enforced).
- **Row blocks.** The global operators are assembled replicated on the CPU
  (scipy CSR, same numba path as serial, including defect pinning and the
  demag anchor DOF); each rank slices its rows and remaps columns into the
  ghost-extended local space `[owned | left plane | right plane]`, asserting
  that no coupling reaches beyond ±1 plane. `ops.DistSparseMat` then performs
  halo exchange + local SpMV under the serial `A @ v` convention.
- **`ops.solve_cg`.** Identical algorithm to the Phase 0 serial solver
  (multi-RHS batching, breakdown freezing, `check_every`), with column dot
  products reduced by `all_reduce(SUM)` — every rank sees identical scalars
  and takes identical branches (convergence, field stepping, stop condition),
  the "all ranks make the same decision" principle.
- **`ops.DistVolumeAverage`.** Rank-local element slice with the exact serial
  per-element quadrature data; field gathering uses corner indices remapped
  into `[owned | right ghost plane]`; integral and volume are reduced in a
  single `all_reduce`. One subtlety inherited from serial: `corner_dofs`
  (DOF ids) index the node-coordinate array when computing `detJ`; the
  distributed class reuses the serial computation verbatim on the local
  element slice so results match bit-for-bit.
- **`micromagnetics.py`.** Mirrors the serial GSPM loop statement for
  statement; per-DOF algebra is rank-local, `sumsq` is all-reduced, the
  anchor RHS entry is zeroed by the rank owning DOF 0, defect DOFs are
  masked locally. VTU/HDF5/hysteresis output is written by rank 0 from
  gathered fields through a serial `VolumeAverage`. ME (magnetoelastic
  coupling) raises `NotImplementedError` (Phase 3); rot111 is supported
  (purely local algebra).

**Validation** (login node, Intel MPI backend, CPU, quick config 16×16×8,
double precision, `doc/prototypes/test_dist_ops.py` + full-simulation runs):

- Unit tests at 2 and 4 ranks: halo planes exact; SpMV on
  `A_demag`/`A1`/`F1`/`Fx` ≤ 2e-15; CG vs serial ≤ 5e-14; multi-RHS CG ≤
  8e-10; volume averages ≤ 2e-17.
- Full hysteresis sweep (58 field values, ~60k LLG steps total) vs serial:
  - **2 ranks:** identical field-step sequence; converged avg_m per field ≤
    2.0e-5; final magnetization field max|diff| = 3.3e-11.
  - **4 ranks:** 57/58 fields ≤ 1e-4; final magnetization field max|diff| =
    1.0e-10. One marginal field (H = −2120 A/m, the slowest-converging state
    of the sweep, ≥1000 LLG steps) converged to a nearby metastable state
    (|Δavg_m| = 4.3e-2) before the trajectories re-merged at the next field —
    the same reduction-order bifurcation sensitivity documented for
    magnum.np.distributed (tolerances there also widen for world_size > 4).

As predicted, wall time at this tiny, latency-bound size is *worse* than
serial (serial 206 s / 8 threads; 2 ranks 353 s; 4 ranks 313 s): each LLG
step issues ~2,600 blocking all-reduces plus ~1,300 two-phase halo
exchanges. This motivated Phase 2; the scaling payoff is expected at large
meshes, mirroring the halo-bound exchange-field result in the paper.

### Phase 2 details (implemented)

Communication-count reductions in the distributed backend:

- **Single-reduction CG** (`ops.solve_cg`): replaced textbook CG with the
  Chronopoulos–Gear variant — per iteration one SpMV (`w = A r`) and **one**
  fused `all_reduce` carrying both dot products (γ = rᵀr, δ = wᵀr), vs two
  reductions before. Same API/semantics (multi-RHS, per-column breakdown
  freezing, `check_every`, explicit final residual check guarding the
  recurrence drift).
- **Comm–compute overlap**: `XSlabPartition.extract_row_block` now splits
  each row block into `A_own` (owned columns) and `A_ghost` (ghost columns);
  `DistSparseMat.matmul` posts the halo exchange, computes `A_own @ x`
  while it is in flight, then adds `A_ghost @ ghosts`. For world_size ≥ 3
  all four P2P ops go in one non-blocking batch; world_size == 2 keeps the
  sequential two-phase form (message-matching ambiguity).
- **Batched solve groups + shared exchanges** (`micromagnetics.py`): the
  `g1n/g2n/g3n` solves (shared `A1`) and `m*starstar` solves (shared `A2`)
  run as single 3-RHS batched CG calls with stacked warm starts; the three
  RHS SpMVs on `m_tilde`, the three derivative SpMVs on `U`, and
  `F1 @ [f1,f2,f3]` each share one halo exchange (`matmul_with_ghosts`).
  Per LLG step: 5 CG solve calls instead of 9, and roughly 4× fewer
  collectives per solve-iteration in the batched groups.

**Phase 2 validation and timings** (same quick-config protocol; note the
login node is shared, ±10% noise):

| Run | Phase 1 | Phase 2 |
|---|---|---|
| serial (8 threads) | 206 s | — |
| 1 rank (distributed path) | — | 303 s |
| 2 ranks × 4 threads | 353 s | 315 s (−11%) |
| 4 ranks × 2 threads | 313 s | 309 s |

- **Variant isolation:** the 1-rank run (zero communication) matches serial
  at all 58 fields (≤ 1.5e-3, final switched state only) — the C–G variant
  introduces only tol-level trajectory differences, no bifurcation.
- 2- and 4-rank runs agree with *each other* to ~1e-5 and match serial
  everywhere except the known marginal zone (H ≈ −2040 A/m flips to the
  nearby metastable state, |Δavg_m| = 3.2e-2, trajectories re-merge) —
  reduction-order sensitivity, same class as Phase 1's 4-rank flip.
- The modest CPU-side gains are expected: on-node Intel-MPI all-reduce
  latency is ~1–2 µs, so collective count is not yet dominant; the same
  reductions target NCCL/multi-node latencies (~20–50 µs) where they
  should matter proportionally more. Two known trade-offs to revisit on
  GPU: batched 3-RHS CG iterates all columns until the slowest converges
  (extra SpMV work when column iteration counts are unbalanced), and the
  1-rank distributed path carries ~45% overhead vs the serial main loop
  (no-op collectives + C-G bookkeeping) — worth profiling on H100 before
  the scaling study.

## 6. Eliminating numba (explored, prototype validated)

The CPU assembly (`assemble_demag`, `assemble_Gauss_Seidel`,
`assemble_elasticity`, jitted helpers in `ShapeHex`/`ShapeTet`) is the only
numba user. Numba pins numpy versions and lags new Python releases — the same
class of installation pain on HPC systems that motivated leaving CuPy.

**"PyTorch's JIT" is not the replacement — vectorization is.** TorchScript
(`torch.jit`) is in maintenance mode, and `torch.compile` cannot accelerate
per-element *scalar Python loops* (it targets tensor programs). The correct
PyTorch-native replacement is batching the element loop into tensor ops:
FEM assembly is embarrassingly element-parallel with identical small dense
structure per element, so the whole loop collapses into a handful of einsums
plus batched `det`/`inv` — the same pattern `VolumeAverage` already uses.

A prototype (`doc/prototypes/hex_assembly_torch.py`, validated by
`doc/prototypes/test_numba_free_assembly.py`) assembles K (stiffness),
M (mass) and Fx/Fy/Fz for Hex meshes this way, using the *same tabulated*
`_N_data`/`_dN_data` shape-function tables, with element chunking to bound
peak memory. Results vs the existing assembly agree to **~1e-15 relative**
(machine precision) for all five operator families, including the anchor-DOF
and degenerate-element (|detJ| < 1e-14) semantics.

Benchmarks (login node, 8 CPU threads, double precision; "numba" timings are
steady-state, i.e. excluding the 3–10 s first-call JIT compilation):

| Mesh (elements) | numba (A + F + GS) | pure-python fallback | vectorized torch | speedup vs numba |
|---|---|---|---|---|
| 16³ | ~1.2 s (post-JIT) | 20.3 s | 0.05–0.2 s | — (JIT dominates) |
| 32³ | 1.20 s | — | 0.37 s | 3.2× |
| 64³ | 13.5 s | — | 3.8 s | 3.5× |
| 96³ | 32.7 s | — | 12.6 s | 2.6× |

Additional wins beyond raw speed:

- **Runs on GPU** (numba path is CPU-only): on an H100 the batched einsum
  assembly would drop to milliseconds, and the COO arrays are born on the
  device — no host→device transfer of multi-GB triplet arrays.
- **Distribution-ready:** per-rank assembly in Phase 1 is just slicing the
  element array before the batched call — no distributed-assembly machinery.
- **Dependency stack shrinks** to torch + numpy + scipy (numba and llvmlite
  gone).

Remaining work to fully drop numba:

1. Port `assemble_elasticity` (24×24 element stiffness / 24×48 coupling via
   the B-matrix — the identical einsum pattern, more index bookkeeping).
2. Replace the remaining *pure-Python* setup loops, which are unrelated to
   numba but dominate setup at scale: `gridHex.std_fem_mesh`'s connectivity
   triple loop and `build_periodic_node_map` (2.8 s at 96³) — both trivially
   vectorizable with numpy meshgrid/`np.unique`.

### Phase 0 details (implemented)

`cupymag_pytorch/solvers/linear_solvers.py` was rewritten so that:

- α, β, `pᵀAp`, `rᵀr` never leave the device; the host receives **one boolean**
  per convergence check instead of two scalars per iteration.
- A `check_every` parameter controls how often that boolean is fetched
  (default 1 preserves the original convergence semantics exactly; larger
  values trade up to `check_every − 1` overshoot iterations for fewer syncs —
  overshoot only lowers the residual further).
- `b` may be 2-D `(n, k)`: columns are solved simultaneously as **batched CG**
  (independent per-column iterates, identical to k separate solves, but with
  shared kernels — and, in Phase 1+, shared collectives). Per-column
  convergence, breakdown freezing (`pᵀAp ≤ 0` freezes that column, mirroring
  the original early `break`), and zero-RHS columns (return exact zeros) are
  handled with device-side masks; no per-iteration host branching.
- The final relative-residual check and the `RuntimeError` on non-convergence
  are preserved (per column).

Behavioural deviations from the original (both benign):
- With `check_every > 1`, converged columns may take a few extra iterations
  before the solver notices (lower residual, slightly different downstream
  numbers). Default is 1.
- Reduction kernels differ (`einsum` vs `torch.dot`), so results can differ at
  floating-point rounding level.
