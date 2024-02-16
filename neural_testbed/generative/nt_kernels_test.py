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
"""Tests for efficient_agent.neural_testbed.generative.nt_kernels."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp

from neural_testbed.generative import nt_kernels


# TODO(author1): move this config update to an explicit initialize function.
jax.config.update('jax_enable_x64', True)


class NtKernelsTest(parameterized.TestCase):

  @parameterized.parameters([[x] for x in range(10)])
  def test_benchmark_kernel(self, seed: int):
    # Generate benchmark kernel
    kernel_fn = nt_kernels.make_benchmark_kernel()
    rng = hk.PRNGSequence(seed)

    # Evaluate at random x in 1D
    x = jax.random.normal(next(rng), [1000, 1])
    kernel = kernel_fn(x, x, 'nngp')
    adjusted_kernel = kernel + 1e-6 * jnp.eye(len(kernel))

    # Check that posterior sample non-nan
    for _ in range(10):
      sample = jax.random.multivariate_normal(
          next(rng), jnp.zeros(len(kernel)), adjusted_kernel)
      assert jnp.all(~jnp.isnan(sample))

  @parameterized.parameters(
      itertools.product(range(10), [1, 10], ['nngp', 'ntk']))
  def test_kernel_matrix(self, seed: int, input_dim: int, method: str):
    """Checks that the kernel matrix is symmetric and positive semi-definite."""
    def is_symmetric(x: jnp.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
      return jnp.allclose(x, x.T, rtol=rtol, atol=atol)

    def is_pos_semi_definite(x: jnp.ndarray):
      return jnp.all(jnp.linalg.eigvals(x) >= -1e-10)

    # Generate benchmark kernel
    kernel_fn = nt_kernels.make_benchmark_kernel()
    rng = hk.PRNGSequence(seed)

    # Evaluate at random x
    x = jax.random.normal(next(rng), [100, input_dim])
    kernel = kernel_fn(x, x, method)

    # Check that the kernel is symmetric, positive semi-definite
    assert is_symmetric(kernel)
    assert is_pos_semi_definite(kernel)


if __name__ == '__main__':
  absltest.main()
