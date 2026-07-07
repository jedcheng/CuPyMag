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

import torch

from cupymag_pytorch.core.parameters import (
    precision,
    s11,
    s12,
    s44,
    sigma11,
    sigma12,
    sigma13,
    sigma22,
    sigma23,
    sigma33,
)
from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.precision_select import get_float_type

float_cp = get_float_type(precision, backend="torch")


def get_Ebar_sigma():
    """
    Calculate the strain from the external stress sigma using the compliance
    tensor (S) for cubic symmetry: epsilon = S : sigma (Voigt notation).

    Returns
    -------
    torch.Tensor: a length-6 strain vector [e11, e22, e33, e12, e23, e13].
    """
    S = torch.zeros((6, 6), dtype=float_cp, device=DEVICE)
    S[0, 0] = S[1, 1] = S[2, 2] = s11
    S[3, 3] = S[4, 4] = S[5, 5] = s44
    S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = s12

    sigma = torch.tensor(
        [sigma11, sigma22, sigma33, sigma12, sigma23, sigma13],
        dtype=float_cp,
        device=DEVICE,
    )

    E_bar_sigma = S @ sigma
    return E_bar_sigma
