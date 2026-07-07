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

from cupymag_pytorch import __version__

logo = rf"""

  /$$$$$$            /$$$$$$$            /$$      /$$
 /$$__  $$          | $$__  $$          | $$$    /$$$
| $$  \__/ /$$   /$$| $$  \ $$ /$$   /$$| $$$$  /$$$$  /$$$$$$   /$$$$$$
| $$      | $$  | $$| $$$$$$$/| $$  | $$| $$ $$/$$ $$ |____  $$ /$$__  $$
| $$      | $$  | $$| $$____/ | $$  | $$| $$  $$$| $$  /$$$$$$$| $$  \ $$
| $$    $$| $$  | $$| $$      | $$  | $$| $$\  $ | $$ /$$__  $$| $$  | $$
|  $$$$$$/|  $$$$$$/| $$      |  $$$$$$$| $$ \/  | $$|  $$$$$$$|  $$$$$$$
 \______/  \______/ |__/       \____  $$|__/     |__/ \_______/ \____  $$
                               /$$  | $$                        /$$  \ $$
                              |  $$$$$$/                       |  $$$$$$/
                               \______/                         \______/

                    PyTorch + Python + Micromagnetics

                          CuPyMag v{__version__} (PyTorch port)
                        Author: Hongyi Guan
                    License: Apache 2.0 License
               https://github.com/Hongyi-Guan/CuPyMag
                    (c) 2025-2026 Hongyi Guan

              If you use this software, please cite:
          1) H. Guan, and A.R. Balakrishna,
          Comput. Phys. Commun. 323, 110093 (2026)
          2) H. Guan, CuPyMag (Version {__version__}), (2026),
          URL: https://github.com/hongyiguan/CuPyMag
"""


def print_logo():
    print(logo)
