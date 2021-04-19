# python3
# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Specific neural tangent kernels."""

import dataclasses
import jax.config
from neural_tangents import stax
from neural_tangents.utils import typing as nt_types
import numpy as np

jax.config.update('jax_enable_x64', True)


@dataclasses.dataclass
class NtkOutput:
  init: nt_types.InitFn
  apply: nt_types.ApplyFn
  kernel: nt_types.KernelFn


def make_benchmark_kernel(input_dim: int = 1) -> nt_types.KernelFn:
  """Generate a benchmark GP kernel for neural testbed."""
  layers = [
      stax.Dense(100, W_std=1, b_std=1 / np.sqrt(input_dim)),
      stax.Sign(),
      stax.Dense(1, W_std=1, b_std=0),
  ]
  _, _, kernel = stax.serial(*layers)
  return kernel


def make_linear_kernel(input_dim: int = 1) -> nt_types.KernelFn:
  """Generate a linear GP kernel for testing putposes."""
  layers = [
      stax.Dense(1, W_std=1, b_std=1 / np.sqrt(input_dim)),
  ]
  _, _, kernel = stax.serial(*layers)
  return kernel
