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
    Return the 8 standard hexahedral shape function values at parametric
    coordinates (r, s, t) on the device (torch tensors).

    Returns
    -------
    torch.Tensor
        An (8, ...) array of shape function values N_i(r, s, t).
    """
    r = torch.as_tensor(r, device=DEVICE)
    s = torch.as_tensor(s, device=DEVICE)
    t = torch.as_tensor(t, device=DEVICE)
    if r.dim() == 0:
        shape = (8,)
    else:
        shape = (8,) + tuple(r.shape)

    N = torch.empty(shape, dtype=float_cp, device=DEVICE)

    N[0] = (1 - r) * (1 - s) * (1 - t) / 8.0
    N[1] = (1 + r) * (1 - s) * (1 - t) / 8.0
    N[2] = (1 + r) * (1 + s) * (1 - t) / 8.0
    N[3] = (1 - r) * (1 + s) * (1 - t) / 8.0
    N[4] = (1 - r) * (1 - s) * (1 + t) / 8.0
    N[5] = (1 + r) * (1 - s) * (1 + t) / 8.0
    N[6] = (1 + r) * (1 + s) * (1 + t) / 8.0
    N[7] = (1 - r) * (1 + s) * (1 + t) / 8.0

    return N


def shape_function_gradients(r, s, t):
    """
    Return the gradient of the 8 hexahedral shape functions w.r.t. the
    parametric coordinates on the device.

    Returns
    -------
    torch.Tensor
        An (8, 3, ...) array of shape function derivatives.
    """
    r = torch.as_tensor(r, device=DEVICE)
    s = torch.as_tensor(s, device=DEVICE)
    t = torch.as_tensor(t, device=DEVICE)
    if r.dim() == 0:
        shape = (8, 3)
    else:
        shape = (8, 3) + tuple(r.shape)

    dN = torch.empty(shape, dtype=float_cp, device=DEVICE)

    dN[0, 0] = -(1 - s) * (1 - t) / 8.0
    dN[0, 1] = -(1 - r) * (1 - t) / 8.0
    dN[0, 2] = -(1 - r) * (1 - s) / 8.0

    dN[1, 0] = +(1 - s) * (1 - t) / 8.0
    dN[1, 1] = -(1 + r) * (1 - t) / 8.0
    dN[1, 2] = -(1 + r) * (1 - s) / 8.0

    dN[2, 0] = +(1 + s) * (1 - t) / 8.0
    dN[2, 1] = +(1 + r) * (1 - t) / 8.0
    dN[2, 2] = -(1 + r) * (1 + s) / 8.0

    dN[3, 0] = -(1 + s) * (1 - t) / 8.0
    dN[3, 1] = +(1 - r) * (1 - t) / 8.0
    dN[3, 2] = -(1 - r) * (1 + s) / 8.0

    dN[4, 0] = -(1 - s) * (1 + t) / 8.0
    dN[4, 1] = -(1 - r) * (1 + t) / 8.0
    dN[4, 2] = +(1 - r) * (1 - s) / 8.0

    dN[5, 0] = +(1 - s) * (1 + t) / 8.0
    dN[5, 1] = -(1 + r) * (1 + t) / 8.0
    dN[5, 2] = +(1 + r) * (1 - s) / 8.0

    dN[6, 0] = +(1 + s) * (1 + t) / 8.0
    dN[6, 1] = +(1 + r) * (1 + t) / 8.0
    dN[6, 2] = +(1 + r) * (1 + s) / 8.0

    dN[7, 0] = -(1 + s) * (1 + t) / 8.0
    dN[7, 1] = +(1 - r) * (1 + t) / 8.0
    dN[7, 2] = +(1 - r) * (1 + s) / 8.0

    return dN


@njit
def element_jacobian(xc, yc, zc, n):
    """
    Compute the 3x3 Jacobian of the reference -> physical map for a
    hexahedral element at Gauss point ``n``.

    Parameters
    ----------
    xc, yc, zc : array_like
        Length-8 nodal coordinates of the element.
    n : int
        Gauss point index.

    Returns
    -------
    numpy.ndarray
        (3, 3) Jacobian matrix.
    """
    dN = get_dN(n)
    J = np.zeros((3, 3), dtype=np.float64)

    for i in range(8):
        J[0, 0] += xc[i] * dN[i, 0]
        J[0, 1] += xc[i] * dN[i, 1]
        J[0, 2] += xc[i] * dN[i, 2]
        J[1, 0] += yc[i] * dN[i, 0]
        J[1, 1] += yc[i] * dN[i, 1]
        J[1, 2] += yc[i] * dN[i, 2]
        J[2, 0] += zc[i] * dN[i, 0]
        J[2, 1] += zc[i] * dN[i, 1]
        J[2, 2] += zc[i] * dN[i, 2]

    return J


@njit
def gauss_quadrature():
    """
    Return the standard 2x2x2 Gauss-Legendre quadrature for linear
    hexahedral elements.

    Returns
    -------
    tuple of numpy.ndarray
        (points (8, 3), weights (8,))
    """
    _gauss_pts_1d = [-0.57735026919, +0.57735026919]
    _gauss_wts_1d = [1.0, 1.0]

    GAUSS_POINTS = []
    GAUSS_WEIGHTS = []

    for rx, wx in zip(_gauss_pts_1d, _gauss_wts_1d):
        for ry, wy in zip(_gauss_pts_1d, _gauss_wts_1d):
            for rz, wz in zip(_gauss_pts_1d, _gauss_wts_1d):
                GAUSS_POINTS.append((rx, ry, rz))
                GAUSS_WEIGHTS.append(wx * wy * wz)

    GAUSS_POINTS = np.array(GAUSS_POINTS, dtype=np.float64)  # (8, 3)
    GAUSS_WEIGHTS = np.array(GAUSS_WEIGHTS, dtype=np.float64)

    return GAUSS_POINTS, GAUSS_WEIGHTS


@njit
def corners_local():
    return np.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [-1.0, +1.0, +1.0],
        ],
        dtype=np.float64,
    )


_dN_data = np.array(
    [
        [
            [-0.311004233964, -0.311004233964, -0.311004233964],
            [0.311004233964, -0.083333333333, -0.083333333333],
            [0.083333333333, 0.083333333333, -0.022329099369],
            [-0.083333333333, 0.311004233964, -0.083333333333],
            [-0.083333333333, -0.083333333333, 0.311004233964],
            [0.083333333333, -0.022329099369, 0.083333333333],
            [0.022329099369, 0.022329099369, 0.022329099369],
            [-0.022329099369, 0.083333333333, 0.083333333333],
        ],
        [
            [-0.083333333333, -0.083333333333, -0.311004233964],
            [0.083333333333, -0.022329099369, -0.083333333333],
            [0.022329099369, 0.022329099369, -0.022329099369],
            [-0.022329099369, 0.083333333333, -0.083333333333],
            [-0.311004233964, -0.311004233964, 0.311004233964],
            [0.311004233964, -0.083333333333, 0.083333333333],
            [0.083333333333, 0.083333333333, 0.022329099369],
            [-0.083333333333, 0.311004233964, 0.083333333333],
        ],
        [
            [-0.083333333333, -0.311004233964, -0.083333333333],
            [0.083333333333, -0.083333333333, -0.022329099369],
            [0.311004233964, 0.083333333333, -0.083333333333],
            [-0.311004233964, 0.311004233964, -0.311004233964],
            [-0.022329099369, -0.083333333333, 0.083333333333],
            [0.022329099369, -0.022329099369, 0.022329099369],
            [0.083333333333, 0.022329099369, 0.083333333333],
            [-0.083333333333, 0.083333333333, 0.311004233964],
        ],
        [
            [-0.022329099369, -0.083333333333, -0.083333333333],
            [0.022329099369, -0.022329099369, -0.022329099369],
            [0.083333333333, 0.022329099369, -0.083333333333],
            [-0.083333333333, 0.083333333333, -0.311004233964],
            [-0.083333333333, -0.311004233964, 0.083333333333],
            [0.083333333333, -0.083333333333, 0.022329099369],
            [0.311004233964, 0.083333333333, 0.083333333333],
            [-0.311004233964, 0.311004233964, 0.311004233964],
        ],
        [
            [-0.311004233964, -0.083333333333, -0.083333333333],
            [0.311004233964, -0.311004233964, -0.311004233964],
            [0.083333333333, 0.311004233964, -0.083333333333],
            [-0.083333333333, 0.083333333333, -0.022329099369],
            [-0.083333333333, -0.022329099369, 0.083333333333],
            [0.083333333333, -0.083333333333, 0.311004233964],
            [0.022329099369, 0.083333333333, 0.083333333333],
            [-0.022329099369, 0.022329099369, 0.022329099369],
        ],
        [
            [-0.083333333333, -0.022329099369, -0.083333333333],
            [0.083333333333, -0.083333333333, -0.311004233964],
            [0.022329099369, 0.083333333333, -0.083333333333],
            [-0.022329099369, 0.022329099369, -0.022329099369],
            [-0.311004233964, -0.083333333333, 0.083333333333],
            [0.311004233964, -0.311004233964, 0.311004233964],
            [0.083333333333, 0.311004233964, 0.083333333333],
            [-0.083333333333, 0.083333333333, 0.022329099369],
        ],
        [
            [-0.083333333333, -0.083333333333, -0.022329099369],
            [0.083333333333, -0.311004233964, -0.083333333333],
            [0.311004233964, 0.311004233964, -0.311004233964],
            [-0.311004233964, 0.083333333333, -0.083333333333],
            [-0.022329099369, -0.022329099369, 0.022329099369],
            [0.022329099369, -0.083333333333, 0.083333333333],
            [0.083333333333, 0.083333333333, 0.311004233964],
            [-0.083333333333, 0.022329099369, 0.083333333333],
        ],
        [
            [-0.022329099369, -0.022329099369, -0.022329099369],
            [0.022329099369, -0.083333333333, -0.083333333333],
            [0.083333333333, 0.083333333333, -0.311004233964],
            [-0.083333333333, 0.022329099369, -0.083333333333],
            [-0.083333333333, -0.083333333333, 0.022329099369],
            [0.083333333333, -0.311004233964, 0.083333333333],
            [0.311004233964, 0.311004233964, 0.311004233964],
            [-0.311004233964, 0.083333333333, 0.083333333333],
        ],
    ]
)

_N_data = np.array(
    [
        [
            0.490562612163,
            0.131445855766,
            0.035220810901,
            0.131445855766,
            0.131445855766,
            0.035220810901,
            0.009437387838,
            0.035220810901,
        ],
        [
            0.131445855766,
            0.035220810901,
            0.009437387838,
            0.035220810901,
            0.490562612163,
            0.131445855766,
            0.035220810901,
            0.131445855766,
        ],
        [
            0.131445855766,
            0.035220810901,
            0.131445855766,
            0.490562612163,
            0.035220810901,
            0.009437387838,
            0.035220810901,
            0.131445855766,
        ],
        [
            0.035220810901,
            0.009437387838,
            0.035220810901,
            0.131445855766,
            0.131445855766,
            0.035220810901,
            0.131445855766,
            0.490562612163,
        ],
        [
            0.131445855766,
            0.490562612163,
            0.131445855766,
            0.035220810901,
            0.035220810901,
            0.131445855766,
            0.035220810901,
            0.009437387838,
        ],
        [
            0.035220810901,
            0.131445855766,
            0.035220810901,
            0.009437387838,
            0.131445855766,
            0.490562612163,
            0.131445855766,
            0.035220810901,
        ],
        [
            0.035220810901,
            0.131445855766,
            0.490562612163,
            0.131445855766,
            0.009437387838,
            0.035220810901,
            0.131445855766,
            0.035220810901,
        ],
        [
            0.009437387838,
            0.035220810901,
            0.131445855766,
            0.035220810901,
            0.035220810901,
            0.131445855766,
            0.490562612163,
            0.131445855766,
        ],
    ]
)


@njit
def get_dN(n):
    """
    Return the 8x3 array of shape-function derivatives at the n-th Gauss
    point (0 <= n <= 7).
    """
    return _dN_data[n]


@njit
def get_N(n):
    """
    Return the length-8 array of shape function values at the n-th Gauss
    point (0 <= n <= 7).
    """
    return _N_data[n]
