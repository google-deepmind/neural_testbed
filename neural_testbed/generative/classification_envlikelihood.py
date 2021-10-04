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

"""Classification-based testbed based around a logit_fn and x_generator."""

from typing import Callable, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from neural_testbed import base as testbed_base
from neural_testbed import likelihood

LogitFn = Callable[[chex.Array], chex.Array]  # x -> logits
XGenerator = Callable[[chex.PRNGKey, int], chex.Array]  # key, num_samples -> x


class ClassificationEnvLikelihood(likelihood.GenerativeDataSampler):
  """Classification-based environment-based inference."""

  def __init__(self,
               logit_fn: LogitFn,
               x_generator: XGenerator,
               num_train: int,
               key: chex.PRNGKey,
               tau: int = 1):

    rng = hk.PRNGSequence(key)

    # Checking the dimensionality of our data coming in.
    x_train = x_generator(next(rng), num_train)
    num_train_x, input_dim = x_train.shape
    assert num_train_x == num_train

    self._logit_fn = logit_fn
    self._tau = tau
    self._x_generator = x_generator
    self._x_train = jnp.array(x_train)
    self._input_dim = input_dim
    self._num_train = num_train

    # Generate environment function across x_train
    train_logits = self._logit_fn(x_train)  # [n_train, n_class]
    self._num_classes = train_logits.shape[-1]  # Obtain from logit_fn.
    chex.assert_shape(train_logits, [self._num_train, self._num_classes])
    self._train_probs = jax.nn.softmax(train_logits)

    # Generate training data.
    def sample_output(probs: chex.Array, key: chex.PRNGKey) -> chex.Array:
      return jax.random.choice(key, self._num_classes, p=probs)
    batch_sampler = lambda p, k: jax.jit(jax.vmap(sample_output))(p, k)[:, None]

    train_keys = jax.random.split(next(rng), self._num_train)
    y_train = batch_sampler(self._train_probs, train_keys)
    self._train_data = testbed_base.Data(x=self._x_train, y=y_train)

    # Generate canonical x_test for DEBUGGING ONLY!!!
    num_test = 1000
    self._x_test = x_generator(next(rng), num_test)
    test_logits = self._logit_fn(self._x_test)  # [n_train, n_class]
    chex.assert_shape(test_logits, [num_test, self._num_classes])
    self._test_probs = jax.nn.softmax(test_logits)

  @property
  def train_data(self) -> testbed_base.Data:
    return self._train_data

  @property
  def test_x(self) -> chex.Array:
    """Canonical test data for debugging only.

    This is not the test data x returned by the test data method.
    """
    return self._x_test

  @property
  def probabilities(self) -> chex.Array:
    """Return probabilities of classes for canonical train and test x.

    Use only for debugging/plotting purposes in conjunction with the test_x
    method. The test_data method does not use the same test_x.
    """
    return jnp.concatenate([self._train_probs, self._test_probs])

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

      x_test = self._x_generator(x_key, self._tau)
      chex.assert_shape(x_test, [self._tau, self._input_dim])
      # Generate environment function across x_test
      test_logits = self._logit_fn(x_test)  # [tau, n_class]
      chex.assert_shape(test_logits, [self._tau, self._num_classes])
      probs = jax.nn.softmax(test_logits)

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
