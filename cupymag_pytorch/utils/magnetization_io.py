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

from pathlib import Path

import h5py
import torch

from cupymag_pytorch.utils.backend import DEVICE


def write_array(array, filename):
    """
    Write a torch tensor to an HDF5 file.

    Parameters
    ----------
    array : torch.Tensor
    filename : str or Path
    """
    filename = Path(filename)
    if filename.suffix != ".h5":
        filename = filename.with_suffix(".h5")

    data = array.detach().cpu().numpy()

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data, compression="gzip", compression_opts=6)
        f.attrs["shape"] = data.shape
        f.attrs["dtype"] = str(data.dtype)


def read_array(filename, nDOF):
    """
    Read a torch tensor from an HDF5 file.

    Returns None (and prints a message) if the file is missing or its
    shape does not match (nDOF, 3).
    """
    if filename is None:
        print("Initial magnetization file not found, start with uniform magnetization.")
        return None

    filename = Path(filename)
    if filename.suffix != ".h5":
        filename = filename.with_suffix(".h5")

    with h5py.File(filename, "r") as f:
        data = f["data"][:]

    if data.shape != (nDOF, 3):
        print(
            f"Initial magnetization shape mismatch: expected ({nDOF}, 3), got {data.shape}."
        )
        print("Start with uniform magnetization.")
        return None

    array = torch.as_tensor(data, device=DEVICE)
    return array


def initialize_m(restart_m, restart, initial_m, nDOF, DefDOF, float_cp):
    """
    Initialize the magnetization field for the simulation.

    Parameters
    ----------
    restart_m : str/Path or None
    restart : bool
    initial_m : array-like or None
    nDOF : int
    DefDOF : torch.Tensor
        Sorted unique global DOF indices of nodes in defect elements.
    float_cp : torch.dtype

    Returns
    -------
    torch.Tensor (nDOF, 3)
    """
    m = read_array(restart_m, nDOF) if restart else None
    if m is None:
        if initial_m is None:
            initial_m = torch.tensor([1.0, 0.0, 0.0], dtype=float_cp, device=DEVICE)
        else:
            initial_m = torch.as_tensor(initial_m, dtype=float_cp, device=DEVICE)
            norm = torch.linalg.vector_norm(initial_m)
            if norm.item() > 0:
                initial_m = initial_m / norm
            else:
                raise ValueError("Initial magnetization vector cannot be zero.")

        m = torch.zeros((nDOF, 3), dtype=float_cp, device=DEVICE)
        m[:, 0] = initial_m[0]
        m[:, 1] = initial_m[1]
        m[:, 2] = initial_m[2]

    m[DefDOF, :] = 0.0

    return m
