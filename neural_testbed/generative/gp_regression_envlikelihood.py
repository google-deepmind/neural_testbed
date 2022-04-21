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
"""GP inference in a regression setting with respect to the environment likelihood."""

from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from neural_tangents._src.utils import typing as nt_types

from neural_testbed import base as testbed_base
from neural_testbed import likelihood


class GPRegressionEnvLikelihood(likelihood.GenerativeDataSampler):
  """GP with gaussian noise output."""

  def __init__(self,
               kernel_fn: nt_types.AnalyticKernelFn,
               x_train: chex.Array,
               x_test: chex.Array,
               key: chex.PRNGKey,
               tau: int = 1,
               noise_std: float = 1,
               kernel_ridge: float = 1e-6,
               ntk: bool = False):

    # Checking the dimensionality of our data coming in.
    num_train, input_dim = x_train.shape
    num_test_x_cache, input_dim_test = x_test.shape
    assert input_dim == input_dim_test

    rng = hk.PRNGSequence(key)
    self._tau = tau
    self._input_dim = input_dim
    self._x_train = jnp.array(x_train)
    self._x_test = jnp.array(x_test)
    self._num_train = num_train
    self._num_test_x_cache = num_test_x_cache
    self._noise_std = noise_std
    self._kernel_ridge = kernel_ridge

    # Generate environment function across combined_x = [x_train, x_test]
    mean = jnp.zeros(num_train + num_test_x_cache)
    get_kernel = 'ntk' if ntk else 'nngp'
    combined_x = jnp.vstack([self._x_train, self._x_test])
    kernel = kernel_fn(combined_x, x2=None, get=get_kernel)
    kernel += kernel_ridge * jnp.eye(len(kernel))
    y_function = jax.random.multivariate_normal(next(rng), mean, kernel)
    chex.assert_shape(y_function, [num_train + num_test_x_cache,])

    # Form the training data
    y_noise = jax.random.normal(next(rng), [num_train, 1]) * noise_std
    y_train = y_function[:num_train, None] + y_noise
    self._train_data = testbed_base.Data(x_train, y_train)
    chex.assert_shape(y_train, [num_train, 1])

    # Form the testing data
    self._y_test_function = y_function[-num_test_x_cache:]
    chex.assert_shape(self._y_test_function, [num_test_x_cache,])

  @property
  def x_test(self) -> chex.Array:
    return self._x_test

  @property
  def train_data(self) -> testbed_base.Data:
    return self._train_data

  def test_data(self, key: chex.PRNGKey) -> Tuple[testbed_base.Data, float]:
    """Generates test data and evaluates log likelihood w.r.t. environment.

    The test data that is output will be of length tau examples.
    We wanted to "pass" tau here... but ran into jax.jit issues.

    Args:
      key: Random number generator key.

    Returns:
      Tuple of data (with tau examples) and log-likelihood under posterior.
    """

    def sample_test_data(key: chex.PRNGKey) -> Tuple[testbed_base.Data, float]:
      x_key, y_key = jax.random.split(key, 2)

      # Sample tau x's from the testing cache for evaluation
      test_x_indices = jax.random.randint(
          x_key, [self._tau], 0, self._num_test_x_cache)
      x_test = self._x_test[test_x_indices]
      chex.assert_shape(x_test, [self._tau, self._input_dim])

      # Sample y_function for the test data
      y_function = self._y_test_function[test_x_indices]
      y_noise = jax.random.normal(y_key, [self._tau, 1]) * self._noise_std
      y_test = y_function[:, None] + y_noise
      data = testbed_base.Data(x_test, y_test)
      chex.assert_shape(y_test, [self._tau, 1])

      # Compute the log likelihood with respect to the environment
      err = y_noise
      chex.assert_shape(err, [self._tau, 1])
      cov = self._noise_std ** 2 * jnp.eye(self._tau)
      chex.assert_shape(cov, [self._tau, self._tau])
      log_likelihood = likelihood.gaussian_log_likelihood(err, cov)
      return data, log_likelihood

    return jax.jit(sample_test_data)(key)
