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

from cupymag_pytorch.core.parameters import precision
from cupymag_pytorch.utils.backend import DEVICE
from cupymag_pytorch.utils.numba_shim import njit
from cupymag_pytorch.utils.precision_select import get_float_type

float_cp = get_float_type(precision, backend="torch")


def shape_functions(r, s, t):
    """
    Return the 4 standard tetrahedral shape function values at parametric
    coordinates (r, s, t) on the device (torch tensors).

    Returns
    -------
    torch.Tensor
        An (4, ...) array of shape function values N_i(r, s, t).
    """
    r = torch.as_tensor(r, device=DEVICE)
    s = torch.as_tensor(s, device=DEVICE)
    t = torch.as_tensor(t, device=DEVICE)
    if r.dim() == 0:
        shape = (4,)
    else:
        shape = (4,) + tuple(r.shape)

    N = torch.empty(shape, dtype=float_cp, device=DEVICE)

    N[0] = 1.0 - r - s - t
    N[1] = r
    N[2] = s
    N[3] = t

    return N


def shape_function_gradients(r, s, t):
    """
    Return the gradient of the 4 tetrahedral shape functions w.r.t. the
    parametric coordinates on the device.

    Returns
    -------
    torch.Tensor
        An (4, 3, ...) array of shape function derivatives.
    """
    r = torch.as_tensor(r, device=DEVICE)
    s = torch.as_tensor(s, device=DEVICE)
    t = torch.as_tensor(t, device=DEVICE)
    if r.dim() == 0:
        shape = (4, 3)
    else:
        shape = (4, 3) + tuple(r.shape)

    dN = torch.empty(shape, dtype=float_cp, device=DEVICE)

    dN[0, 0] = torch.full_like(r, -1.0)
    dN[0, 1] = torch.full_like(s, -1.0)
    dN[0, 2] = torch.full_like(t, -1.0)

    dN[1, 0] = torch.ones_like(r)
    dN[1, 1] = torch.zeros_like(s)
    dN[1, 2] = torch.zeros_like(t)

    dN[2, 0] = torch.zeros_like(r)
    dN[2, 1] = torch.ones_like(s)
    dN[2, 2] = torch.zeros_like(t)

    dN[3, 0] = torch.zeros_like(r)
    dN[3, 1] = torch.zeros_like(s)
    dN[3, 2] = torch.ones_like(t)

    return dN


@njit
def element_jacobian(xc, yc, zc, n):
    """
    Compute the 3x3 Jacobian of the reference -> physical map for a linear
    tetrahedral element. ``n`` is unused (the Jacobian is constant).

    Returns
    -------
    numpy.ndarray
        (3, 3) Jacobian matrix.
    """
    J = np.zeros((3, 3), dtype=np.float64)

    J[:, 0] = xc[1:] - xc[0]
    J[:, 1] = yc[1:] - yc[0]
    J[:, 2] = zc[1:] - zc[0]

    return J


@njit
def gauss_quadrature():
    """
    Return the standard Gauss-Legendre quadrature for linear tetrahedral
    elements.

    Returns
    -------
    tuple of numpy.ndarray
        (points (4, 3) in barycentric (r, s, t), weights (4,))
    """
    a = 0.58541020
    b = 0.13819660

    GAUSS_POINTS = np.array(
        [[b, b, b], [a, b, b], [b, a, b], [b, b, a]], dtype=np.float64
    )

    GAUSS_WEIGHTS = np.ones(4, dtype=np.float64) / 24.0

    return GAUSS_POINTS, GAUSS_WEIGHTS


@njit
def corners_local():
    """Return the corners of the reference tetrahedral element."""
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


_N_data = np.array(
    [
        [0.5854102, 0.1381966, 0.1381966, 0.1381966],
        [0.1381966, 0.5854102, 0.1381966, 0.1381966],
        [0.1381966, 0.1381966, 0.5854102, 0.1381966],
        [0.1381966, 0.1381966, 0.1381966, 0.5854102],
    ]
)

_dN_data = np.array(
    [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


@njit
def get_dN(n):
    """
    Return the 4x3 array of shape-function derivatives (constant over the
    element, so ``n`` is ignored).
    """
    return _dN_data


@njit
def get_N(n):
    """
    Return the length-4 array of shape function values at the n-th Gauss
    point (0 <= n <= 3).
    """
    return _N_data[n]
