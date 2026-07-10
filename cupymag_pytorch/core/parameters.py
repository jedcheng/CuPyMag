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

import os
import sys
from pathlib import Path

import numpy as np
import torch

from cupymag_pytorch.core.config_loader import ConfigLoader
from cupymag_pytorch.core.constants import GAMMA, MU0
from cupymag_pytorch.utils.backend import DEVICE


def _validate_parameters(ms, ld, timestep, AA_raw, alpha_damp):
    if ms <= 0:
        raise ValueError("Saturation magnetization must be positive")
    if ld <= 0:
        raise ValueError("Characteristic length must be positive")
    if timestep <= 0:
        raise ValueError("Timestep must be positive")
    if AA_raw <= 0:
        raise ValueError("Exchange constant must be positive")
    if alpha_damp < 0:
        raise ValueError("Damping constant alpha must be non-negative")


def _safe_divide(numerator, denominator, operation_name):
    if abs(denominator) < 1e-30:
        raise ValueError(f"Division by zero in {operation_name}: {denominator}")
    return numerator / denominator


# ----------------------------------------------------------------------
# Resolve the configuration file path.
#   * sys.argv[1] takes priority (matches the original behaviour, where
#     parameters.py is imported at the top of cli.py before argparse runs)
#   * otherwise the CUPYMAG_CONFIG_PATH env var
#   * otherwise 'config.yaml' in the current directory
# ----------------------------------------------------------------------
config_path = None
if len(sys.argv) > 1:
    config_path = Path(sys.argv[1]).resolve()
else:
    config_path = Path(os.environ.get("CUPYMAG_CONFIG_PATH", "config.yaml")).resolve()

if not config_path.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {config_path}\n"
        f"Please create a config.yaml file or copy one from the examples/ directory."
    )

config_loader = ConfigLoader(config_path)
config = config_loader.config

# Load parameters
# Physics parameters
physics = config["physics"]

ms = physics["magnetic"]["ms"]
AA_raw = physics["magnetic"]["exchange_stiffness"]
K1_raw = physics["magnetic"]["anisotropy_k1"]

c11_raw = physics["elastic"]["c11"]
c12_raw = physics["elastic"]["c12"]
c44_raw = physics["elastic"]["c44"]

lambda100 = physics["magnetostriction"]["lambda100"]
lambda111 = physics["magnetostriction"]["lambda111"]

demag = physics["demagnetization"]
N_x = demag["n_x"]
N_y = demag["n_y"]
N_z = demag["n_z"]

field = physics["external_field"]
Hext1_raw = field["hext1"]
dHext1_raw = field["dhext1"]
Hext2_raw = field.get("hext2", 0.0)
dHext2_raw = field.get("dhext2", 0.0)
Hext3_raw = field.get("hext3", 0.0)
dHext3_raw = field.get("dhext3", 0.0)

external_stress = physics["external_stress"]
sigma11 = external_stress.get("sigma11", 0.0)
sigma22 = external_stress.get("sigma22", 0.0)
sigma33 = external_stress.get("sigma33", 0.0)
sigma12 = external_stress.get("sigma12", 0.0)
sigma23 = external_stress.get("sigma23", 0.0)
sigma13 = external_stress.get("sigma13", 0.0)

rot111 = physics["rotation_111"]
ME = physics["magnetoelastic_coupling"]
ld = physics["characteristic_length"]

# Grid parameters
grid = config["grid"]
grid_type = grid["type"]
defect_center = grid.get("defect_center")
if defect_center is not None:
    defect_center = torch.as_tensor(defect_center, dtype=torch.float64, device=DEVICE)

if grid_type == "Hex":
    cubic_mesh = grid["cubic_mesh"]
    Nx = cubic_mesh.get("nx")
    Ny = cubic_mesh.get("ny")
    Nz = cubic_mesh.get("nz")
    Ndx = cubic_mesh.get("ndx")
    Ndy = cubic_mesh.get("ndy")
    Ndz = cubic_mesh.get("ndz")
elif grid_type == "Tet":
    tet_mesh = grid["tet_mesh"]
    mesh_file = tet_mesh["mesh_file"]
    defect_shape = tet_mesh["defect_shape"]
    shape_params = np.array(tet_mesh["defect_parameters"])
    rotation_matrix = tet_mesh.get("rotation_matrix", None)
    if rotation_matrix is not None:
        shape_params = (shape_params, np.array(rotation_matrix))
else:
    raise NotImplementedError(f"Grid type '{grid_type}' is not supported.")

# Simulation parameters
sim = config["simulation"]
precision = sim["precision"]

llg = sim["llg"]
timestep = llg["timestep"]
alpha_damp = llg["alpha_damping"]
LLG_accuracy_factor = llg["llg_accuracy_factor"]

solver = sim["solver"]
tol = solver["tolerance"]
maxiter = solver["max_iterations"]
use_init_factor = solver["initial_factor"]
# Fetch the CG convergence flag from the device only every N iterations
# (up to N-1 overshoot iterations past convergence; reduces host syncs,
# which matters on GPU).
cg_check_every = int(solver.get("check_every", 1))
# CG preconditioner: "none" or "jacobi" (inverse-diagonal scaling;
# communication-free, mainly useful on non-uniform/Tet meshes).
cg_preconditioner = str(solver.get("preconditioner", "none")).lower()
if cg_preconditioner not in ("none", "jacobi"):
    raise ValueError(f"Unknown solver.preconditioner: {cg_preconditioner}")

restart = sim.get("restart", False)
restart_m = sim.get("restart_magnetization", None)

if restart == False:
    initial_magnetization = sim.get("initial_magnetization", [1.0, 0.0, 0.0])
else:
    initial_magnetization = None

stop_condition = sim.get("stop_condition", {})
stop_component = stop_condition.get("component", None)
if stop_component is None:
    stop_direction = stop_condition.get("direction", None)
    if stop_direction is None:
        stop_direction = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float64, device=DEVICE
        )
    else:
        stop_direction = torch.as_tensor(
            stop_direction, dtype=torch.float64, device=DEVICE
        )
        norm = torch.linalg.vector_norm(stop_direction)
        if norm.item() > 0:
            stop_direction = stop_direction / norm
        else:
            raise ValueError("Direction vector cannot be zero.")
else:
    stop_direction = torch.zeros(3, dtype=torch.float64, device=DEVICE)
    stop_direction[stop_component] = 1.0


stop_value = stop_condition.get("value", -0.95)
if not (0 > stop_value > -1):
    stop_value = -0.95

# Output settings
output = config.get("output", {})
write_m = output.get("write_magnetization", True)
output_dir = config_loader.get_output_dir()
save_frequency = int(output.get("save_frequency", 1000))
vtu_alpha = output.get("vtu_blend_alpha", 0.05)

_validate_parameters(ms, ld, timestep, AA_raw, alpha_damp)

# Characteristic magnetic pressure
psi0 = MU0 * ms**2

# Normalized parameters
K1 = _safe_divide(K1_raw, psi0, "K1 normalization")
AA = _safe_divide(AA_raw, psi0, "AA normalization")
c11 = _safe_divide(c11_raw, psi0, "c11 normalization")
c12 = _safe_divide(c12_raw, psi0, "c12 normalization")
c44 = _safe_divide(c44_raw, psi0, "c44 normalization")

# Normalized external stress
sigma11 = _safe_divide(sigma11, psi0, "sigma11 normalization")
sigma22 = _safe_divide(sigma22, psi0, "sigma22 normalization")
sigma33 = _safe_divide(sigma33, psi0, "sigma33 normalization")
sigma12 = _safe_divide(sigma12, psi0, "sigma12 normalization")
sigma23 = _safe_divide(sigma23, psi0, "sigma23 normalization")
sigma13 = _safe_divide(sigma13, psi0, "sigma13 normalization")

# Compliance tensor components
s11 = (c11 + c12) / (pow(c11, 2) + c11 * c12 - 2 * pow(c12, 2))
s12 = -c12 / (pow(c11, 2) + c11 * c12 - 2 * pow(c12, 2))
s44 = 1 / c44

# Elastic coupling constants
elasC1 = 2 * c44 * (1.5 * lambda111**2) - (c11 - c12) * (1.5 * lambda100) ** 2
elasC2 = -1.5 * lambda100 * (c11 - c12)
elasC3 = -6.0 * lambda111 * c44

# Time stepping
dt = _safe_divide(timestep * (MU0 * GAMMA) * ms, (1 + alpha_damp**2), "Time step dt")

# Exchange parameters
astar = _safe_divide(2 * AA, ld**2, "Exchange parameter")
alpha = astar * dt

# External field (normalized)
Hext1 = _safe_divide(Hext1_raw, ms, "Hext1 normalization")
dHext1 = _safe_divide(dHext1_raw, ms, "dHext1 normalization")
Hext2 = _safe_divide(Hext2_raw, ms, "Hext2 normalization")
dHext2 = _safe_divide(dHext2_raw, ms, "dHext2 normalization")
Hext3 = _safe_divide(Hext3_raw, ms, "Hext3 normalization")
dHext3 = _safe_divide(dHext3_raw, ms, "dHext3 normalization")
