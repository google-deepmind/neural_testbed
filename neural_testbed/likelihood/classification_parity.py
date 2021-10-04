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

"""Calculating classification likelihood based on the parity check algorithm.

WARNING: THIS IS EXPERIMENTAL CODE AND NOT YET AT GOLD QUALITY.
"""
# TODO(author2): sort out the code quality here.

import dataclasses

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.likelihood import base as likelihood_base
from neural_testbed.likelihood import utils
import typing_extensions


class LogProbabilityEstimator(typing_extensions.Protocol):

  def __call__(self, y_target: chex.Array, y_samples: chex.Array) -> float:
    """Estimate the log probability of y_target derived from y_samples.

    Args:
      y_target: vector of size dimension=d
      y_samples: matrix of [num_samples=k, d]

    Returns:
      Log probability estimate.
    """


def _convert_to_binary(x: chex.Array, axis=None) -> chex.Array:
  """Convert an array of nonnegative ints to binaries."""
  # TODO(lxlu): See if we can optimize memory usage.
  return jnp.unpackbits(x.astype(jnp.uint8), axis=axis)


def make_parity_check_estimator(
    num_features: int,
    key: chex.PRNGKey,
    num_classes: int,
    projection_method: str = 'binary',
) -> LogProbabilityEstimator:
  """Factory method to create parity_check estimator according to note."""
  rng = hk.PRNGSequence(key)

  def binary_projection(y_target: chex.Array,
                        y_samples: chex.Array,
                        key: chex.PRNGKey) -> chex.Array:
    """Binary projection that converts labels to binary encodings."""
    y_target_binary = _convert_to_binary(y_target)
    y_samples_binary = _convert_to_binary(y_samples, axis=1)

    num_samples, binary_length = y_samples_binary.shape
    assert len(y_target_binary) == binary_length

    projection_matrix = jax.random.bernoulli(
        key, p=0.5, shape=[num_features, binary_length],
    ).astype(jnp.int32)

    projected_samples = jnp.einsum(
        'li,ki->lk', projection_matrix, y_samples_binary) % 2
    chex.assert_shape(projected_samples, [num_features, num_samples])
    return projected_samples

  def modulo_projection(y_target: chex.Array,
                        y_samples: chex.Array,
                        key: chex.PRNGKey) -> chex.Array:
    """Modulo random projection by number of classes."""
    y_dimension = len(y_target)  # Currently d in the writeup
    projection_matrix = jax.random.bernoulli(
        key, p=0.5, shape=[num_features, y_dimension],
    ).astype(jnp.int32)

    num_samples, sample_dim = y_samples.shape
    assert sample_dim == y_dimension

    projected_samples = jnp.einsum(
        'li,ki->lk', projection_matrix, y_samples) % num_classes
    chex.assert_shape(projected_samples, [num_features, num_samples])
    return projected_samples

  if projection_method == 'binary':
    compute_projected_samples = binary_projection
    num_value_per_feat = 2
    assert num_classes <= 256  # binary encoding limited to 8 bits
  elif projection_method == 'mod_num_classes':
    compute_projected_samples = modulo_projection
    num_value_per_feat = num_classes
  else:
    raise ValueError('Unrecognized projection method.')

  all_features = utils.enumerate_all_features(num_features, num_value_per_feat)

  def estimator(y_target: chex.Array, y_samples: chex.Array) -> float:
    """Estimate the log likelihood of y_target derived from y_samples."""

    num_samples, y_dimension = y_samples.shape

    # 1. Compute features
    projected_samples = compute_projected_samples(
        y_target, y_samples, next(rng))

    # 2. Compute the p_hat estimator
    def count_equal_samples(single_f: chex.Array) -> chex.Array:
      batched_equal = jax.vmap(jnp.array_equal, in_axes=(1, None))
      return jnp.sum(batched_equal(projected_samples, single_f))

    # Output the batched computation over all codes f
    p_hat = jax.lax.map(count_equal_samples, all_features) / num_samples
    chex.assert_shape(p_hat, [num_value_per_feat ** num_features])

    # 3. Compute q_hat estimator
    def single_q_hat(single_f: chex.Array, i: int):
      # Check for the rows where samples match f
      batched_equal = jax.vmap(jnp.array_equal, in_axes=(1, None))
      valid_masks = batched_equal(
          projected_samples, single_f).astype(jnp.float32)
      chex.assert_shape(valid_masks, [num_samples])

      # Check whether the sampled X matches the target X in component
      matching_x = jnp.equal(y_samples[:, i], y_target[i]).astype(jnp.float32)
      chex.assert_shape(matching_x, [num_samples])

      # Divide with buffer so 0/0 gives 1
      numerator = jnp.sum(matching_x * valid_masks)
      denominator = jnp.sum(valid_masks)
      return (numerator + 1e-6) / (denominator + 1e-6)

    dim_q_hat = jax.vmap(single_q_hat, in_axes=(None, 0))

    # Output the batched computation over all codes f and all dimensions i
    map_fn = lambda x: dim_q_hat(x, jnp.arange(y_dimension))
    all_qs = jax.lax.map(map_fn, all_features)
    chex.assert_shape(all_qs, [num_value_per_feat ** num_features, y_dimension])

    # 4. Combine the approximation ll = log(sum_f p_hat(f) * prod_i q_i(f))
    log_prod_qs = jnp.sum(jnp.log(all_qs), axis=1)
    chex.assert_shape(log_prod_qs, [num_value_per_feat ** num_features])
    base = jnp.max(log_prod_qs)
    ll = jnp.log(jnp.sum(p_hat * jnp.exp(log_prod_qs - base))) + base

    return ll

  return jax.jit(estimator)


@dataclasses.dataclass
class CategoricalParityCheckKL(likelihood_base.SampleBasedKL):
  """Evaluates KL according to categorical model, sampling X and output Y.

  This approach samples an (x, y) output from the enn and data sampler and uses
  this to estimate the KL divergence.
  """
  num_test_seeds: int  # Number of (x, y) datasets with len(y)=tau
  num_enn_samples: int  # Number of samples from ENN environment models
  num_enn_realizations: int  # Number of y_hat drawn from each ENN sample
  estimator: LogProbabilityEstimator
  key: chex.PRNGKey
  num_classes: int  # Purely for shape checking

  def __call__(
      self,
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: likelihood_base.GenerativeDataSampler,
  ) -> testbed_base.ENNQuality:
    """Evaluates KL according to categorical model."""

    def generate_y_sample_from_enn_class_probs(
        class_probs: chex.Array, key: chex.Array) -> chex.Array:
      """Samples a single length-tau realization from class_probs."""
      tau, num_classes = class_probs.shape
      assert num_classes == self.num_classes
      def sample_output(k: chex.PRNGKey, p: chex.Array) -> chex.Array:
        """Samples a single output for a single key, for single class probs."""
        return jax.random.choice(k, self.num_classes, p=p)
      y_keys = jax.random.split(key, tau)
      y_sample = jax.vmap(sample_output)(y_keys, class_probs)
      chex.assert_shape(y_sample, [tau])
      return y_sample

    def kl_estimate(key: chex.PRNGKey) -> float:
      """Computes KL estimate on a single instance of test data."""
      data_key, sample_key = jax.random.split(key)

      # Generate the data
      data, true_ll = data_sampler.test_data(data_key)
      tau = data.x.shape[0]

      # Sample realizations in y from the ENN class probabilities
      def generate_multi_y_samples(key: chex.Array) -> chex.Array:
        """Probabilities for one ENN and all of the data."""

        enn_key, sampler_key = jax.random.split(key)

        # Sample the class probabilities
        enn_samples = enn_sampler(data.x, enn_key)
        probs = jax.nn.softmax(enn_samples)
        chex.assert_shape(probs, [tau, self.num_classes])

        # Function to sample single y output (length tau) per key
        sample_y = lambda k: generate_y_sample_from_enn_class_probs(probs, k)

        # lax.map samples num_enn_realizations outputs, each with different key
        sample_keys = jax.random.split(sampler_key, self.num_enn_realizations)
        y_samples = jax.lax.map(sample_y, sample_keys)
        chex.assert_shape(y_samples, [self.num_enn_realizations, tau])
        return y_samples

      # generate_multi_samples for each num_enn_samples
      sample_keys = jax.random.split(sample_key, self.num_enn_samples)
      # Attempt to use jitted code, but if the enn_sampler is not able to
      # jax.jit then run without jit.
      try:
        y_samples = jax.lax.map(generate_multi_y_samples, sample_keys)
      except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
        # TODO(author1): replace with proper logging.
        print(f'Was not able to run enn_sampler inside jit due to: {e}')
        y_samples = jnp.array([generate_multi_y_samples(k)
                               for k in sample_keys])

      chex.assert_shape(
          y_samples,
          [self.num_enn_samples, self.num_enn_realizations, tau]
      )
      y_samples = jnp.reshape(y_samples, [-1, tau])
      y_target = jnp.reshape(data.y, [tau])

      # Estimate the log likelihood via the estimator
      model_ll = self.estimator(y_target, y_samples)
      return true_ll - model_ll

    kl_keys = jax.random.split(self.key, self.num_test_seeds)
    # Attempt to use jitted code.
    try:
      kl_estimates = jax.lax.map(kl_estimate, kl_keys)
    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
      # TODO(author1): replace with proper logging.
      print(f'Failed to run kl estimator with jit due to: {e}.')
      kl_estimates = jnp.array([kl_estimate(k) for k in kl_keys])

    return utils.parse_kl_estimates(kl_estimates)

