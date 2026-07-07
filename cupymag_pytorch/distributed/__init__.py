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

"""Distributed (multi-device) backend for CuPyMag-PyTorch.

Phase 1 implementation (see ``doc/distributed_approach.md``): x-slab
row-block partition of the periodic Hex FEM operators, ring halo
exchange, distributed CG, and rank-0 I/O. Launch with::

    mpirun -n 4 python -m cupymag_pytorch.distributed config.yaml --backend=mpi
    torchrun --nproc_per_node=4 -m cupymag_pytorch.distributed config.yaml --backend=nccl
"""
