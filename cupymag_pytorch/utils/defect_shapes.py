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

from abc import ABC, abstractmethod

import numpy as np
import torch


class DefectShape(ABC):
    """Abstract base class for defect shapes."""

    _shape_registry = {}

    def __init__(self, defect_shape, shape_params):
        self.defect_shape = defect_shape
        self.shape_params = shape_params

        if defect_shape not in self._shape_registry:
            supported_shapes = list(self._shape_registry.keys())
            raise NotImplementedError(
                f"Defect shape '{defect_shape}' is not supported. "
                f"Supported shapes: {supported_shapes}"
            )

        self._shape_class = self._shape_registry[defect_shape]

    @classmethod
    def register_shape(cls, name, shape_class):
        cls._shape_registry[name] = shape_class

    @classmethod
    def get_supported_shapes(cls):
        return list(cls._shape_registry.keys())

    def get_shape_function(self):
        shape_instance = self._shape_class(self.shape_params)
        return shape_instance.create_shape_function()

    @abstractmethod
    def create_shape_function(self):
        pass


class SphereShape(DefectShape):
    """Sphere defect shape."""

    def __init__(self, shape_params):
        self.R = float(shape_params)
        self.R2 = self.R * self.R

    def create_shape_function(self):
        R2 = self.R2

        def shape_fn(rel_coords):
            if rel_coords.ndim == 1:
                return torch.dot(rel_coords, rel_coords) < R2
            else:
                return torch.sum(rel_coords**2, dim=1) < R2

        return shape_fn


class EllipsoidShape(DefectShape):
    """Axis-aligned ellipsoid defect shape."""

    def __init__(self, shape_params):
        self.a, self.b, self.c = shape_params
        self.inv2 = torch.tensor(
            [1.0 / self.a**2, 1.0 / self.b**2, 1.0 / self.c**2], dtype=torch.float64
        )

    def create_shape_function(self):
        inv2 = self.inv2

        def shape_fn(rel_coords):
            rel_coords = rel_coords.to(inv2.dtype)
            if rel_coords.ndim == 1:
                return torch.sum(inv2 * rel_coords**2) < 1.0
            else:
                return torch.sum(rel_coords**2 * inv2[None, :], dim=1) < 1.0

        return shape_fn


class RotatedEllipsoidShape(DefectShape):
    """Rotated ellipsoid defect shape."""

    def __init__(self, shape_params):
        abc, rotation_matrix = shape_params
        self.a, self.b, self.c = abc
        self.R = torch.as_tensor(np.asarray(rotation_matrix), dtype=torch.float64)
        inv_diag = torch.diag(
            torch.tensor(
                [1.0 / self.a**2, 1.0 / self.b**2, 1.0 / self.c**2], dtype=torch.float64
            )
        )
        self.Q = self.R @ inv_diag @ self.R.T

    def create_shape_function(self):
        Q = self.Q

        def shape_fn(rel_coords):
            rel_coords = rel_coords.to(Q.dtype)
            if rel_coords.ndim == 1:
                return rel_coords @ Q @ rel_coords < 1.0
            else:
                temp = rel_coords @ Q
                return torch.sum(temp * rel_coords, dim=1) < 1.0

        return shape_fn


DefectShape.register_shape("Sphere", SphereShape)
DefectShape.register_shape("Ellipsoid", EllipsoidShape)
DefectShape.register_shape("RotatedEllipsoid", RotatedEllipsoidShape)


def get_defect_shape_function(defect_shape, shape_params):
    """
    Get a defect shape function.

    Parameters
    ----------
    defect_shape : str
        Name of the defect shape.
    shape_params : various
        Parameters specific to the shape type.

    Returns
    -------
    function
        shape_fn(rel_coords) -> bool or array of bool
    """
    shape_class = DefectShape._shape_registry[defect_shape]
    shape_instance = shape_class(shape_params)
    return shape_instance.create_shape_function()
