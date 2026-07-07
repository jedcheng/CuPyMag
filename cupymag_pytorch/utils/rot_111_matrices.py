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

import numpy as np
import torch

from cupymag_pytorch.core.constants import *
from cupymag_pytorch.utils.backend import DEVICE


def get_R_matrix(backend="numpy", dtype=None):
    """
    Generate the 3x3 R matrix for the [100] -> [111] rotation.

    Parameters
    ----------
    backend: str
        'numpy' or 'torch'
    dtype:
        data type for the matrix

    Returns
    -------
    array: 3x3 R matrix
    """
    if backend == "numpy":
        xp = np
        if dtype is None:
            dtype = np.float64
    elif backend == "torch":
        xp = torch
        if dtype is None:
            dtype = torch.float64
    else:
        raise ValueError("Backend must be 'numpy' or 'torch'")

    R = xp.array(
        [
            [inv_sqrt3, inv_sqrt3, inv_sqrt3],
            [-inv_sqrt2, inv_sqrt2, 0.0],
            [-inv_sqrt6, -inv_sqrt6, 2 * inv_sqrt6],
        ],
        dtype=dtype,
    )

    if backend == "torch":
        R = R.to(DEVICE)
    return R


def get_M_matrix(backend="numpy", dtype=None):
    """
    Generate the 6x6 M matrix for the [100] -> [111] rotation.

    Parameters
    ----------
    backend: str
        'numpy' or 'torch'
    dtype:
        data type for the matrix

    Returns
    -------
    array: 6x6 M matrix
    """
    if backend == "numpy":
        xp = np
        if dtype is None:
            dtype = np.float64
    elif backend == "torch":
        xp = torch
        if dtype is None:
            dtype = torch.float64
    else:
        raise ValueError("Backend must be 'numpy' or 'torch'")

    M = xp.array(
        [
            [1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3],
            [1 / 2, 1 / 2, 0.0, 0.0, 0.0, -1.0],
            [1 / 6, 1 / 6, 2 / 3, -2 / 3, -2 / 3, 1 / 3],
            [sqrt3 / 6, -sqrt3 / 6, 0.0, sqrt3 / 3, -sqrt3 / 3, 0.0],
            [-sqrt2 / 6, -sqrt2 / 6, sqrt2 / 3, sqrt2 / 6, sqrt2 / 6, -sqrt2 / 3],
            [-inv_sqrt6, inv_sqrt6, 0.0, inv_sqrt6, -inv_sqrt6, 0.0],
        ],
        dtype=dtype,
    )

    if backend == "torch":
        M = M.to(DEVICE)
    return M
