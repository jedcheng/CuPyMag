# Copyright (c) 2025-2026 Hongyi Guan
# This file is part of CuPyMag
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

from typing import Any

import yaml


def print_system_info_summary(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    print(" " * 20 + "System Information Summary")
    print("=" * 64)

    def print_section(obj, indent=0):
        spaces = "  " * indent

        def format_value(value: Any) -> str:
            if isinstance(value, float):
                if abs(value) < 1e-20:
                    return "0.0"
                elif abs(value) >= 1e6 or abs(value) <= 1e-4:
                    return f"{value:.2e}"
                elif abs(value) >= 1000:
                    return f"{value:,.0f}"
                else:
                    return f"{value:.4g}"
            return str(value)

        if isinstance(obj, dict):
            for key, value in obj.items():
                key += str(":")
                if isinstance(value, (dict, list)):
                    print(f"{spaces}{key}")
                    print_section(value, indent + 1)
                else:
                    width = 30 - len(spaces)
                    print(f"{spaces}{key:<{width}} {format_value(value)}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    print(f"{spaces}[{i}]:")
                    print_section(item, indent + 1)
                else:
                    print(f"{spaces}- {item}")

    print_section(data)
    print("=" * 64)
