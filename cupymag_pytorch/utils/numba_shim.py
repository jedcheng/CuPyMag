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

"""Optional Numba shim.

The original CuPyMag uses ``@njit``/``@jitclass`` to JIT-compile the
CPU-side finite-element assembly. Numba is independent of CuPy and works
with prebuilt wheels (no CUDA compilation), so it is the recommended way
to keep assembly fast.

If Numba is not installed (or its experimental ``jitclass`` is missing),
this module provides transparent no-op fallbacks so the code still runs
in pure NumPy/Python, just slower. A ``spec`` dictionary passed to the
fallback ``jitclass`` is ignored.
"""

try:
    from numba import float64, int32, int64, njit  # type: ignore

    try:
        from numba.experimental import jitclass  # type: ignore

        HAS_NUMBA = True
    except Exception:
        HAS_NUMBA = False
        raise
except Exception:
    HAS_NUMBA = False

    class _DummyNumbaType:
        """Stand-in for a Numba type that tolerates ``[:]``/``[:, :]`` slicing."""

        def __getitem__(self, item):
            return self

    float64 = _DummyNumbaType()
    int32 = _DummyNumbaType()
    int64 = _DummyNumbaType()

    def njit(*args, **kwargs):
        # Support both @njit and @njit(parallel=True, ...) usage.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn):
            return fn

        return deco

    def jitclass(spec):
        def deco(cls):
            return cls

        return deco
