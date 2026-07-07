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


def get_float_type(precision: str, backend: str = "numpy"):
    """
    Return the appropriate float type for the given backend and precision.

    Parameters
    ----------
    precision : str
        Either "SP" (single precision) or "DP" (double precision).
    backend : str
        One of "numpy", "torch", or "numba". The legacy "cupy" option is
        retained for backward compatibility but is unused by this port.

    Returns
    -------
    type
        Corresponding float type.

    Raises
    ------
    ValueError
        If precision or backend is invalid.
    """
    if precision not in ("SP", "DP"):
        raise ValueError(f"Invalid precision '{precision}'. Choose 'SP' or 'DP'.")

    single = precision == "SP"

    if backend == "numpy":
        import numpy as xp

        return xp.float32 if single else xp.float64
    elif backend == "torch":
        import torch

        return torch.float32 if single else torch.float64
    elif backend == "numba":
        from numba import float32, float64

        return float32 if single else float64
    elif backend == "cupy":
        try:
            import cupy as xp
        except ImportError:
            raise ImportError(
                "Backend 'cupy' could not be imported. The PyTorch port does not use CuPy."
            )
        return xp.float32 if single else xp.float64
    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose 'numpy', 'torch', or 'numba'."
        )
