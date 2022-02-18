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

from neural_tangents import stax
from neural_tangents._src.utils import typing as nt_types
import numpy as np
import typing_extensions


class KernelCtor(typing_extensions.Protocol):
  """Interface for generating a kernel for a given input dimension."""

  def __call__(self, input_dim: int) -> nt_types.AnalyticKernelFn:
    """Generates a kernel for a given input dimension."""


@dataclasses.dataclass
class MLPKernelCtor(KernelCtor):
  """Generates a GP kernel corresponding to an infinitely-wide MLP."""
  num_hidden_layers: int
  activation: nt_types.InternalLayer

  def __post_init__(self):
    assert self.num_hidden_layers >= 1, 'Must have at least one hidden layer.'

  def __call__(self, input_dim: int = 1) -> nt_types.AnalyticKernelFn:
    """Generates a kernel for a given input dimension."""
    limit_width = 50  # Implementation detail of neural_testbed, unused.
    layers = [
        stax.Dense(limit_width, W_std=1, b_std=1 / np.sqrt(input_dim))
    ]
    for _ in range(self.num_hidden_layers - 1):
      layers.append(self.activation)
      layers.append(stax.Dense(limit_width, W_std=1, b_std=0))
    layers.append(self.activation)
    layers.append(stax.Dense(1, W_std=1, b_std=0))
    _, _, kernel = stax.serial(*layers)
    return kernel


def make_benchmark_kernel(input_dim: int = 1) -> nt_types.AnalyticKernelFn:
  """Creates the benchmark kernel used in leaderboard = 2-layer ReLU."""
  kernel_ctor = MLPKernelCtor(num_hidden_layers=2, activation=stax.Relu())
  return kernel_ctor(input_dim)


def make_linear_kernel(input_dim: int = 1) -> nt_types.AnalyticKernelFn:
  """Generate a linear GP kernel for testing putposes."""
  layers = [
      stax.Dense(1, W_std=1, b_std=1 / np.sqrt(input_dim)),
  ]
  _, _, kernel = stax.serial(*layers)
  return kernel
