#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
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

source $SSD/venv/cupymag_gpu/bin/activate
export PYTHONPATH=/fast/pj24001684/pytorch_mpi/CuPyMag

cd $PJM_O_WORKDIR

CONFIG_QUICK=/fast/pj24001684/pytorch_mpi/CuPyMag/examples/example_config_quick.yaml

# Correctness smoke test at 2 and 4 GPUs (compare hysteresis.txt vs a
# serial run of the same config).
torchrun --standalone --nproc_per_node=2 -m cupymag_pytorch.distributed \
    $CONFIG_QUICK --backend=nccl
torchrun --standalone --nproc_per_node=4 -m cupymag_pytorch.distributed \
    $CONFIG_QUICK --backend=nccl

# Scaling workload: create a 64x64x32 variant of the quick config first
# (nx/ny/nz + a fresh output directory), then:
# for P in 1 2 4; do
#     torchrun --standalone --nproc_per_node=$P -m cupymag_pytorch.distributed \
#         config_bench_64.yaml --backend=nccl
# done
