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

import argparse
import os
import sys
from pathlib import Path

from cupymag_pytorch import __version__


def main():
    parser = argparse.ArgumentParser(
        description="CuPyMag (PyTorch port): GPU-Accelerated Finite-Element Micromagnetics with Magnetostriction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          cupymag-pytorch example_config.yaml
          python -m cupymag_pytorch example_config.yaml
          python -m cupymag_pytorch.cli example_config.yaml
          cupymag-pytorch --help
        """,
    )

    parser.add_argument(
        "config_file",
        nargs="?",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--version", action="version", version=f"CuPyMag v{__version__} (PyTorch port)"
    )

    args = parser.parse_args()

    config_path = Path(args.config_file).resolve()
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found.")
        print(
            f"Please create a config.yaml file or copy one from the examples/ directory."
        )
        return 1

    try:
        os.environ["CUPYMAG_CONFIG_PATH"] = str(config_path)

        # Imported lazily so the config path is picked up correctly, and so
        # that --help / --version work without requiring a valid config.
        from cupymag_pytorch.core.Micromagnetics import main as run_simulation

        run_simulation()
        return 0

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
