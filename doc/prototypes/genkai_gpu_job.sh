#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=8
#PJM -L elapse=1:00:00
#PJM -j
#
# Genkai GPU-node scaling benchmark for the CuPyMag distributed backend
# (NCCL). TEMPLATE — verify rscgrp / gpu directives against the current
# Genkai user guide and your project quota before submitting:
#   - GPU resource group name (b-batch assumed here, 4x H100 per node)
#   - whether "#PJM -L gpu=4" (or vnode-core) is required on this system
#
# Unlike the CPU/MPI path, the GPU path needs NO source-built torch:
# stock CUDA wheels ship NCCL. One-time setup on a GPU node or login node:
#
#   python3.11 -m venv $SSD/venv/cupymag_gpu
#   source $SSD/venv/cupymag_gpu/bin/activate
#   pip install torch numpy scipy meshio h5py pyyaml pandas numba
#
# The benchmark metric is the per-field-step "Time for LLG calculation
# used" lines printed by the main loop; compare across world sizes at
# fixed mesh size (quick config = correctness smoke test; the 64x64x32
# config below is the scaling workload — takes several minutes/field).

module purge
module load cuda/12.2.2

source $SSD/venv/magnumnp_v2_dist_gpu/bin/activate
export WKDIR=/fast/pj24001684/pytorch_mpi/CuPyMag

cd $WKDIR

CONFIG_QUICK=$WKDIR/examples/example_config_quick.yaml

CUDA_VISIBLE_DEVICES=0 python -m cupymag_pytorch --config $CONFIG_QUICK 


# Correctness smoke test at 2 and 4 GPUs (compare hysteresis.txt vs a
# serial run of the same config).
torchrun --standalone --nproc_per_node=2 -m cupymag_pytorch.distributed \
    $CONFIG_QUICK --backend=nccl
torchrun --standalone --nproc_per_node=4 -m cupymag_pytorch.distributed \
    $CONFIG_QUICK --backend=nccl
torchrun --standalone --nproc_per_node=8 -m cupymag_pytorch.distributed \
    $CONFIG_QUICK --backend=nccl

# Scaling workload: 64x64x32 (131k DOFs), with solver.check_every: 8 to
# cut per-iteration GPU->host syncs. Compare the per-field "Time for LLG
# calculation used" lines across P; let the elapse limit cut the sweep.
# CONFIG_BENCH=$WKDIR/doc/prototypes/config_bench_64.yaml
# for P in 1 2 4 8; do
#     torchrun --standalone --nproc_per_node=$P -m cupymag_pytorch.distributed \
#         $CONFIG_BENCH --backend=nccl
# done
#
# Also worth one measurement: the quick config with check_every: 8 at
# 1 GPU, to quantify how much of the 329 s single-GPU smoke-test time was
# host-sync latency.
