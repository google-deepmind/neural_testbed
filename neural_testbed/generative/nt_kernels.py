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
from typing import Any, List, Optional, Tuple, TypeVar, Union

from jax import random
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
import numpy as np
from typing_extensions import Protocol


T = TypeVar('T')
PyTree = Any
NTTree = Union[List[T], Tuple[T, ...], T]


Shapes = NTTree[Tuple[int, ...]]
"""A shape - a tuple of integers, or an :class:`NTTree` of such tuples.
"""


# Layer Definition.


class InitFn(Protocol):
  """A type alias for initialization functions.

  Initialization functions construct parameters for neural networks given a
  random key and an input shape. Specifically, they produce a tuple giving the
  output shape and a PyTree of parameters.
  """

  def __call__(
      self,
      rng: random.KeyArray,
      input_shape: Shapes,
      **kwargs
  ) -> Tuple[Shapes, PyTree]:
    ...


class ApplyFn(Protocol):
  """A type alias for apply functions.

  Apply functions do computations with finite-width neural networks. They are
  functions that take a PyTree of parameters and an array of inputs and produce
  an array of outputs.
  """

  def __call__(
      self,
      params: PyTree,
      inputs: NTTree[np.ndarray],
      *args,
      **kwargs
  ) -> NTTree[np.ndarray]:
    ...


KernelOrInput = Union[NTTree[nt.Kernel], NTTree[np.ndarray]]


Get = Union[Tuple[str, ...], str, None]


class LayerKernelFn(Protocol):
  """A type alias for pure kernel functions.

  A pure kernel function takes a PyTree of Kernel object(s) and produces a
  PyTree of Kernel object(s). These functions are used to define new layer
  types.
  """

  def __call__(
      self,
      k: NTTree[nt.Kernel]
  ) -> NTTree[nt.Kernel]:
    ...


class AnalyticKernelFn(Protocol):
  """A type alias for analytic kernel functions.

  A kernel function that computes an analytic kernel. Takes either a
  :class:`~neural_tangents.Kernel` or :class:`jax.numpy.ndarray` inputs and a
  `get` argument that specifies what quantities should be computed by the
  kernel. Returns either a :class:`~neural_tangents.Kernel` object or
  :class:`jax.numpy.ndarray`-s for kernels specified by `get`.
  """

  def __call__(
      self,
      x1: KernelOrInput,
      x2: Optional[NTTree[np.ndarray]] = None,
      get: Get = None,
      **kwargs
  ) -> Union[NTTree[nt.Kernel], NTTree[np.ndarray]]:
    ...


InternalLayer = Tuple[InitFn, ApplyFn, LayerKernelFn]


class KernelCtor(Protocol):
  """Interface for generating a kernel for a given input dimension."""

  def __call__(self, input_dim: int) -> AnalyticKernelFn:
    """Generates a kernel for a given input dimension."""


@dataclasses.dataclass
class MLPKernelCtor(KernelCtor):
  """Generates a GP kernel corresponding to an infinitely-wide MLP."""
  num_hidden_layers: int
  activation: InternalLayer

  def __post_init__(self):
    assert self.num_hidden_layers >= 1, 'Must have at least one hidden layer.'

  def __call__(self, input_dim: int = 1) -> AnalyticKernelFn:
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


def make_benchmark_kernel(input_dim: int = 1) -> AnalyticKernelFn:
  """Creates the benchmark kernel used in leaderboard = 2-layer ReLU."""
  kernel_ctor = MLPKernelCtor(num_hidden_layers=2, activation=stax.Relu())
  return kernel_ctor(input_dim)


def make_linear_kernel(input_dim: int = 1) -> AnalyticKernelFn:
  """Generate a linear GP kernel for testing putposes."""
  layers = [
      stax.Dense(1, W_std=1, b_std=1 / np.sqrt(input_dim)),
  ]
  _, _, kernel = stax.serial(*layers)
  return kernel
