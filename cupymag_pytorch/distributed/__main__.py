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

"""Entry point for the distributed simulator.

    mpirun -n 4 python -m cupymag_pytorch.distributed config.yaml --backend=mpi
    torchrun --nproc_per_node=4 -m cupymag_pytorch.distributed config.yaml --backend=nccl
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="CuPyMag (PyTorch port) — distributed multi-device simulator"
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--backend",
        choices=["mpi", "nccl", "gloo"],
        default="mpi",
        help="torch.distributed backend (default: mpi)",
    )
    args = parser.parse_args()

    config_path = Path(args.config_file).resolve()
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found.")
        return 1

    # parameters.py reads sys.argv[1] as the config path; hand it the
    # config via the environment instead and clear our own arguments.
    os.environ["CUPYMAG_CONFIG_PATH"] = str(config_path)
    sys.argv = sys.argv[:1]

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend=args.backend)

    if args.backend == "nccl" and torch.cuda.is_available():
        device_id = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

    try:
        # Imported lazily so the config path is picked up correctly.
        from cupymag_pytorch.distributed.micromagnetics import main as run

        run()
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
