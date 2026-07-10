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
| **3a** | Distributed magnetoelasticity (ME). | **done** |
| **3b** | Tet path (general partition), Jacobi preconditioning, parallel VTU output. | **done** |
| **next** | GPU/multi-node scaling study (config + job script ready); optional METIS ordering; CUDA-graph capture of the CG body. | planned |

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

#### Why the batched solve groups work (explainer)

Both Phase 2 batching tricks exploit the same fact: **collective latency is
paid per message, not per byte**. One LLG step solves

```
1. U          ← CG on A_demag
2. g1n        ← CG on A1  ┐
3. g2n        ← CG on A1  ├─ independent, same matrix, different RHS
4. g3n        ← CG on A1  ┘
5. g1star     ← CG on A1     (needs g1n,g2n,g3n via m1star)
6. g2star     ← CG on A1     (needs g1star via m2star)
7. m1starstar ← CG on A2  ┐
8. m2starstar ← CG on A2  ├─ independent, same matrix, different RHS
9. m3starstar ← CG on A2  ┘
```

Solves 2–4 (and 7–9) are the x/y/z components of the same projection
substep: same matrix, three right-hand sides, no dependency between them.
Only `g1star`/`g2star` form a genuine sequential chain and stay single.

*Trick 1 — batched CG.* `solve_cg(A1, B)` with `B` of shape (n, 3) runs
three mathematically independent per-column CGs (batched, not block CG —
iterates identical to separate solves; α, β, γ, δ are per-column). Per
iteration: the SpMV reads the matrix once for all three columns (SpMV is
memory-bound, matrix traffic dominates → extra columns nearly free), the
halo exchange moves one (plane, 3) block instead of three messages, and the
fused γ/δ reduction carries a (2, 3) tensor — one collective serving three
solves. Warm starts stack as the (n, 3) solution of the previous step.
Trade-off: the batched call returns when *all* columns converge, so early
finishers keep iterating until the slowest is done — extra compute traded
for fewer messages (wins when latency dominates, loses when compute does —
part of the 1-rank overhead).

*Trick 2 — shared exchanges.* `Fx @ U`, `Fy @ U`, `Fz @ U` apply different
matrices to the *same vector*, and the ghost planes fetched by the halo
exchange belong to the vector, not the matrix. So the exchange runs once
(`part.get_ghosts`) and each matrix consumes the same ghost block via
`matmul_with_ghosts` — six exchanges become two for the `m_tilde` and `U`
SpMV groups, and `F1 @ [f1, f2, f3]` is one exchange as a 3-column SpMV.

Net per LLG step: 9 → 5 CG calls; 6 → 1 all-reduces per iteration of a
solve triple; 9 → 3 exchanges for the RHS/derivative SpMVs.

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

### Phase 3a details (implemented): distributed magnetoelasticity

The elasticity system's layouts make x-slab distribution reuse the scalar
machinery directly:

- `A_el` (3n × 3n) is **component-major** (`row = c·n + node`): a 3×3 grid
  of n×n blocks, each with the scalar hex sparsity. `ops.DistBlockMat`
  extracts the nine row blocks with the existing remap/split and performs
  the block SpMV with **one** halo exchange of the (n_own, 3) displacement
  field; the `A @ v` interface keeps the serial flattened component-major
  vector so the coupled system runs through `solve_cg` as a single column.
- `F_el` (3n × 6n) columns are **node-major with stride 6** (Voigt
  components per node), so node slabs are still contiguous column ranges;
  `XSlabPartition.remap_strided(6)` generalizes the column remap and
  `ops.DistFMat` maps the local (n_own, 6) spontaneous strain `E0` to the
  RHS with one exchange. The three anchor DOFs (rigid-body modes) are
  applied inside the replicated assembler and come along for free.
- `ops.compute_E_from_u_dist` mirrors `ComputeDerivatives.compute_E_from_u`
  (plain and rot111) with one shared exchange for all nine derivative
  SpMVs; strain de-meaning reuses `DistVolumeAverage` on the (n_own, 6)
  field. The E_bar algebra is replicated scalars, hence rank-local.
- The distributed main loop now carries the full serial ME branch
  (`E0 → b_el → u → E → E_tilde + E_bar`), the elastic terms in the LLG
  effective field, and `llg_min_step = 10`; rot111+ME is supported.

**Validation** (quick config with `magnetoelastic_coupling: true`, Intel
MPI, CPU): unit tests at 2/4 ranks — `A_el`/`F_el` SpMV ≤ 8e-16 relative,
CG on `A_el` vs serial ≤ 3e-16, `compute_E_from_u` (plain and rotated)
≤ 5e-16. Full hysteresis sweep vs serial ME reference: 59/59 field steps,
converged avg_m ≤ 1.2e-4 at every field (no metastable flips — the
cleanest multi-rank sweep so far), final switched state ≤ 3.1e-4.
Wall time at this tiny size remains slower than serial (735 s vs 289 s at
2 ranks; the 6144-unknown elasticity solve is latency-bound like the
rest — GPU/multi-node is where the design targets).

A PJM job-script template for the Genkai GPU-node NCCL scaling benchmark
is at `doc/prototypes/genkai_gpu_job.sh` (the GPU path needs no
source-built torch — stock CUDA wheels ship NCCL).

### Phase 3b details (implemented)

- **Jacobi-preconditioned CG** (`solver.preconditioner: "jacobi"`): the
  solver `M` parameter is now the inverse diagonal — standard PCG in the
  serial solver, preconditioned Chronopoulos–Gear in the distributed one
  (still a single fused `all_reduce`, three stacked dot products including
  the true-residual convergence norm). Communication-free; the diagonal is
  near-constant on uniform Hex meshes, so its value is mainly on
  non-uniform/Tet meshes. Quick-config smoke run matches the
  unpreconditioned baseline to 2e-11.
- **Parallel VTU** (`output.parallel_vtu: true`): each rank writes its
  element slice as a `.vtu` piece through the ghost-extended local index
  (no gather); rank 0 emits the `.pvtu` index. Validated bit-identical to
  the gather-based write of the same trajectory. The restart `.h5` stays
  gathered (global DOF order, restart-compatible at any rank count).
- **Tet meshes — `GeneralPartition`.** DOFs are renumbered by a
  lexicographic coordinate sort (x slowest — the unstructured analogue of
  the x-slab layout) so each rank owns a contiguous range of the permuted
  numbering; ghosts are the element-adjacency neighbours of owned DOFs
  (exactly the FEM coupling stencil) exchanged via per-neighbour index
  lists — one message per directed pair, so a single non-blocking batch is
  safe at any world size, unlike the two-phase plane exchange. Elements
  are assigned by the min-permuted-corner rule (guaranteeing all corners
  lie in own + ghost); gathers return the original DOF order, so output
  and restart files are identical layouts to serial. Everything downstream
  (`DistSparseMat`, block elasticity, volume average, parallel VTU) works
  through the shared partition interface; a METIS ordering could later
  replace the coordinate sort by supplying a different permutation.
  Note: like the serial assembly, positively-oriented CTETRA connectivity
  is assumed (negative-volume tets make the operators indefinite).
- **Validation** (structured Kuhn tet mesh, 1152 DOFs / 6912 tets,
  generated by `doc/prototypes/make_tet_mesh.py`): all unit tests
  (SpMV / CG / PCG / elasticity operators / volume averages) at machine
  precision with 2 and 4 ranks; full hysteresis sweep vs serial Tet —
  57/57 fields, converged avg_m ≤ 2.4e-4, no metastable flips.

### GPU smoke test (H100 node, NCCL — quick config, 2048 DOFs)

First interactive run on a Genkai GPU node (`doc/prototypes/log.txt`,
rscgrp c-batch, cuda/12.2.2):

| GPUs | total time | final avg m1 |
|---|---|---|
| 1 (serial main) | 329 s | −0.6978300335 |
| 2 | 671 s | −0.6989541644 |
| 4 | 630 s | −0.6989541644 |
| 8 | 684 s | −0.6989541644 |

- **Correctness:** the 2/4/8-GPU runs agree with each other to ~1e-10 and
  their convergence traces match to 7 digits at every checkpoint —
  essentially reproducible distributed trajectories across world sizes.
  The ~1.1e-3 offset vs 1 GPU is the serial-main (textbook CG) vs
  distributed (C–G variant) trajectory difference in the switched state,
  same as on CPU. This validates the full NCCL code path, including the
  8-rank case (2 planes/rank, the enforced minimum).
- **Performance:** at 2048 DOFs everything is launch/collective latency —
  one H100 loses to 8 CPU cores (329 vs 207 s), and multi-GPU is ~2×
  single GPU, flat in P (256 DOFs/rank at P=8 amortizes nothing). The
  quick config is a correctness smoke test only; the scaling signal needs
  `doc/prototypes/config_bench_64.yaml` (131k DOFs) or larger.
- **Cheap next win:** the CG loop still performs one GPU→host sync per
  iteration for the convergence flag. `solver.check_every: N` in the YAML
  (plumbed through both mains; default 1) fetches it every N iterations
  instead — at ~10–20 µs per sync × thousands of iterations per LLG step,
  this is likely a large fraction of the single-GPU 329 s.

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
