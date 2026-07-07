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


import warnings
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please copy an example config from the examples/ directory."
            )

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {self.config_path}: {e}")

        self._validate_config(config)
        return config

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure and values"""
        required_sections = ["physics", "grid", "simulation", "output"]

        for section in required_sections:
            if section not in config:
                warnings.warn(
                    f"Warning: Missing required section: {section}, necessary for micromagnetic simulations.",
                    UserWarning,
                )

    def get(self, key_path: str, default=None):
        """Get nested configuration value using dot notation"""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})

    def get_output_dir(self):
        output = self.get("output", {}) or {}
        directory = output.get("directory")
        if directory is None:
            return self.config_path.parent.resolve()
        dir_path = Path(directory)
        if dir_path.is_absolute():
            return dir_path
        return (self.config_path.parent / dir_path).resolve()
