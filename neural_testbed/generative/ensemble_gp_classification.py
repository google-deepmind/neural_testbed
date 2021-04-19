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

"""Ensemble-based GP inference in a classification setting.

Our model is that the true posterior distribution over GPs can be effectively
approximated over num_models=N sampled functions, for N sufficiently large.
If you take this ensemble of N models as the "true" posterior, then you can
compute *exact* posterior inference over this ensemble.

This implementation focuses on test_distribution_order=1, which only compares
the quality of our distributional approximation in terms a singleton piece of
test data.

In an idealized mathematical formulation, we would generate a random (x_i, y_i)
for test data at each seed. However, in the GP setting we find it more
convenient to pre-register a finite x_test, from which we will then sample the
realizations for KL testing.
"""

from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from neural_tangents.utils import typing as nt_types

from neural_testbed import base as testbed_base
from neural_testbed import likelihood


class GPClassificationEnsemble(likelihood.GenerativeDataSampler):
  """GP with softmax output, neural_tangent kernel, ensemble-based inference.

  Our model is that the true posterior distribution over GPs can be effectively
  approximated by num_models=N sampled models, for N sufficiently large.
  If you take this ensemble of N models as the "true" posterior, then you can
  compute *exact* posterior inference over this ensemble.

  We use a single sample of test data to estimate the KL quality. To do this
  efficiently, we first generate num_test_x_cache x_i which can be used for
  test data. When a sample of test data is desired, we first sample an x
  uniformly from this test_x_cache, and then sample an label y corresponding
  to that value.
  """

  def __init__(self,
               kernel_fn: nt_types.KernelFn,
               x_train: chex.Array,
               x_test: chex.Array,
               num_classes: int = 2,
               temperature: float = 1,
               num_models: int = 10_000,
               seed: int = 1,
               kernel_ridge: float = 1e-6,
               ntk: bool = False):

    # Checking the dimensionality of our data coming in.
    num_train, input_dim = x_train.shape
    num_test_x_cache, input_dim_test = x_test.shape
    assert input_dim == input_dim_test

    rng = hk.PRNGSequence(seed)
    self._input_dim = input_dim
    self._x_train = jnp.array(x_train)
    self._x_test = jnp.array(x_test)
    self._num_train = num_train
    self._num_test_x_cache = num_test_x_cache
    self._num_models = num_models
    self._num_classes = num_classes

    # Generate true functions across combined_x = [x_train, x_test]
    mean = jnp.zeros(num_train + num_test_x_cache)
    get_kernel = 'ntk' if ntk else 'nngp'
    combined_x = jnp.vstack([self._x_train, self._x_test])
    kernel = kernel_fn(combined_x, x2=None, get=get_kernel)
    # TODO(author2): investigate need for kernel_ridge.
    kernel += kernel_ridge * jnp.eye(len(kernel))

    def sample_single_model_probs(key: chex.PRNGKey) -> chex.Array:
      """Samples a single model in the ensemble's class probabilities."""
      sample_logit = lambda x: jax.random.multivariate_normal(x, mean, kernel)
      sample_all_class_logits = jax.vmap(sample_logit, out_axes=1)
      logits = sample_all_class_logits(jax.random.split(key, num_classes))
      return jax.nn.softmax(logits / temperature)  # [data, classes]

    sample_probabilities = jax.jit(jax.vmap(sample_single_model_probs))
    keys = jax.random.split(next(rng), num_models)

    # Class probabilities for each ensemble model, for each data point.
    self._probabilities = sample_probabilities(keys)  # [models, data, classes]
    self._true_probs = self._probabilities[0]  # [data, classes]

    chex.assert_shape(self._probabilities, [
        self._num_models,
        self._num_train + self._num_test_x_cache,
        self._num_classes,
    ])
    chex.assert_shape(self._true_probs, [
        self._num_train + self._num_test_x_cache,
        self._num_classes,
    ])

    # Generate training data.
    def sample_output(probs: chex.Array, key: chex.PRNGKey) -> chex.Array:
      return jax.random.choice(key, num_classes, p=probs)

    train_probs = self._true_probs[:num_train]
    train_seeds = jax.random.split(next(rng), num_train)
    batched_sample = jax.jit(jax.vmap(sample_output))
    y_train = batched_sample(train_probs, train_seeds)[:, None]
    self._train_data = testbed_base.Data(x=self._x_train, y=y_train)

    # Compute the posterior
    self._posterior = compute_posterior(
        self._probabilities[:, :num_train, :], y_train)

  @property
  def train_data(self) -> testbed_base.Data:
    return self._train_data

  @property
  def posterior(self) -> chex.Array:
    return self._posterior

  @property
  def probabilities(self) -> chex.Array:
    return self._probabilities

  def test_data(self, seed: int) -> Tuple[testbed_base.Data, float]:
    """Generates test data and evaluates log likelihood under posterior.

    Note that for num_test_seeds <= num_test_x_cache then you will not sample
    the same cached x twice.

    Args:
      seed: Integer random seed, first selects x round-robin from pre-generated
        num_test_x_cache. Then, samples a label y for this x.

    Returns:
      Tuple of data (single x,y pair) and log-likelihood under posterior.
    """

    def single_test_data(seed: int) -> Tuple[testbed_base.Data, float]:
      probs_key, y_key = jax.random.split(jax.random.PRNGKey(seed), 2)

      # Be careful about the shapes of these arrays:
      chex.assert_shape(self._probabilities, [
          self._num_models,
          self._num_train + self._num_test_x_cache,
          self._num_classes,
      ])
      chex.assert_shape(
          self._x_test, [self._num_test_x_cache, self._input_dim])

      # Sample an x from the testing cache for evaluation.
      test_x_index = seed % self._num_test_x_cache
      probs_x_index = self._num_train + test_x_index

      # Sample an ensemble probability from the posterior over probabilities
      sampled_probs_index = jax.random.choice(
          probs_key, self._num_models, p=jnp.squeeze(self._posterior))

      # Class probabilities are evaluated at that x, for that ensemble element
      probs = self._probabilities[sampled_probs_index, probs_x_index, :]

      # Sample a singleton y label for this given x according to probs.
      x_test = self._x_test[test_x_index, :]
      y_test = jax.random.choice(
          y_key, self._num_classes, shape=(1, 1), p=probs)
      data = testbed_base.Data(x=x_test[None, :], y=y_test)

      # Checking the shape of our data
      num_x, input_dim = data.x.shape
      assert num_x == 1
      assert input_dim == self._input_dim
      assert data.y.shape == (1, 1)

      # Compute the log-likelihood at this data.
      batched_ll = jax.vmap(
          likelihood.categorical_log_likelihood, in_axes=[0, None])
      test_probs = self._probabilities[:, probs_x_index, :]
      sampled_ll = batched_ll(test_probs[:, None, :], y_test)
      log_likelihood = jnp.log(jnp.dot(self._posterior, jnp.exp(sampled_ll)))
      chex.assert_shape(sampled_ll, (self._num_models,))
      chex.assert_shape(sampled_ll, self._posterior.shape)

      return data, log_likelihood

    return jax.jit(single_test_data)(seed)


def compute_posterior(
    probabilities: chex.Array, y_train: chex.Array) -> chex.Array:
  """Compute the multinomial posterior sampled from candidate probabilities.

  Args:
    probabilities: Array of candidate probabilities of shape
      [num_models, num_train, num_classes].
    y_train: y label of training data = [num_train, 1]

  Returns:
    Posterior probability of each candidate model.
  """
  num_models, num_train, unused_num_class_fn = probabilities.shape
  chex.assert_shape(y_train, (num_train, 1))

  batched_ll = jax.jit(jax.vmap(
      likelihood.categorical_log_likelihood, in_axes=[0, None]))
  log_likelihood = batched_ll(probabilities, y_train)
  unnormalized_posterior = jnp.exp(log_likelihood)

  assert len(unnormalized_posterior) == num_models
  return unnormalized_posterior / jnp.sum(unnormalized_posterior)
