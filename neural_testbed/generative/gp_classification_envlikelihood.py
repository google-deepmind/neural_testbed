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

"""GP inference in a classification setting with respect to the environment likelihood."""

from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from neural_tangents.utils import typing as nt_types

from neural_testbed import base as testbed_base
from neural_testbed import likelihood


class GPClassificationEnvLikelihood(likelihood.GenerativeDataSampler):
  """GP with softmax output, neural_tangent kernel, environment-based inference."""

  def __init__(self,
               kernel_fn: nt_types.KernelFn,
               x_train: chex.Array,
               x_test: chex.Array,
               key: chex.PRNGKey,
               tau: int = 1,
               num_classes: int = 2,
               temperature: float = 1,
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
    self._num_classes = num_classes

    # Generate environment function across combined_x = [x_train, x_test]
    mean = jnp.zeros(num_train + num_test_x_cache)
    get_kernel = 'ntk' if ntk else 'nngp'
    combined_x = jnp.vstack([self._x_train, self._x_test])
    kernel = kernel_fn(combined_x, x2=None, get=get_kernel)
    kernel += kernel_ridge * jnp.eye(len(kernel))

    def sample_environment_probs(key: chex.PRNGKey) -> chex.Array:
      """Samples environment class probabilities for the data."""
      sample_logit = lambda x: jax.random.multivariate_normal(x, mean, kernel)
      sample_all_class_logits = jax.vmap(sample_logit, out_axes=1)
      logits = sample_all_class_logits(jax.random.split(key, num_classes))
      return jax.nn.softmax(logits / temperature)  # [data, classes]

    # Class probabilities for each data point.
    self._probabilities = sample_environment_probs(next(rng))  # [data, classes]

    chex.assert_shape(self._probabilities, [
        self._num_train + self._num_test_x_cache,
        self._num_classes,
    ])

    # Generate training data.
    def sample_output(probs: chex.Array, key: chex.PRNGKey) -> chex.Array:
      return jax.random.choice(key, num_classes, p=probs)

    train_probs = self._probabilities[:num_train]
    train_keys = jax.random.split(next(rng), num_train)
    batched_sample = jax.jit(jax.vmap(sample_output))
    y_train = batched_sample(train_probs, train_keys)[:, None]
    self._train_data = testbed_base.Data(x=self._x_train, y=y_train)

    self._test_probs = self._probabilities[num_train:]

  @property
  def train_data(self) -> testbed_base.Data:
    return self._train_data

  @property
  def test_x(self) -> chex.Array:
    return self._x_test

  @property
  def probabilities(self) -> chex.Array:
    return self._probabilities

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

      # Be careful about the shapes of these arrays:
      chex.assert_shape(
          self._test_probs, [self._num_test_x_cache, self._num_classes])
      chex.assert_shape(
          self._x_test, [self._num_test_x_cache, self._input_dim])

      # Sample tau x's from the testing cache for evaluation.
      test_x_indices = jax.random.randint(
          x_key, [self._tau], 0, self._num_test_x_cache)

      # For these x indices, find class probabilities.
      probs = self._test_probs[test_x_indices, :]
      chex.assert_shape(probs, [self._tau, self._num_classes])

      # For these x indices, find the corresponding x test.
      x_test = self._x_test[test_x_indices, :]
      chex.assert_shape(x_test, [self._tau, self._input_dim])

      def sample_output(key: chex.PRNGKey, p: chex.Array) -> chex.Array:
        """Samples a single output for a single key, for single class probs."""
        return jax.random.choice(key, self._num_classes, shape=(1,), p=p)
      y_keys = jax.random.split(y_key, self._tau)
      y_test = jax.vmap(sample_output)(y_keys, probs)
      data = testbed_base.Data(x=x_test, y=y_test)
      chex.assert_shape(data.x, [self._tau, self._input_dim])
      chex.assert_shape(data.y, [self._tau, 1])

      # Compute the log likelihood with respect to the environment
      log_likelihood = likelihood.categorical_log_likelihood(probs, y_test)
      return data, log_likelihood

    return jax.jit(sample_test_data)(key)
