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

__version__ = "0.9.1-pt"
__author__ = "Hongyi Guan"
__email__ = "hongyi_guan@ucsb.edu"
__license__ = "Apache 2.0"

import torch

# CuPyMag was GPU-only (CuPy). The PyTorch port will run on CUDA when available
# and fall back to CPU otherwise. The chosen device is exposed for downstream use.
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
else:
    _DEVICE = torch.device("cpu")
    import warnings

    warnings.warn(
        "CUDA is not available. CuPyMag-PyTorch will run on the CPU, which is much slower.",
        RuntimeWarning,
    )
