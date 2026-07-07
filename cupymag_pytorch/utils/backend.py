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

"""Backend helpers for the PyTorch port of CuPyMag.

This module centralizes the device selection (CUDA when available, CPU
otherwise) and provides small conversion helpers that replace the CuPy
idioms used throughout the original code base.
"""

import numpy as np
import torch

from cupymag_pytorch import _DEVICE

# The single device used for all GPU/torch tensors in this port.
DEVICE = _DEVICE


def to_np(x):
    """Convert a torch tensor (or pass-through array) to a NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_tensor(x, dtype=None):
    """Convert a python/numpy sequence to a torch tensor on DEVICE."""
    if isinstance(x, torch.Tensor):
        t = x.to(DEVICE)
    else:
        t = torch.as_tensor(np.asarray(x), device=DEVICE)
    if dtype is not None:
        t = t.to(dtype)
    return t
