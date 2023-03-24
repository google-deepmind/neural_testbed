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

"""Utility functions for calculating likelihood."""

import dataclasses

import chex
from enn.metrics import base as metrics_base
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.likelihood import base as likelihood_base


def gaussian_log_likelihood(err: chex.Array,
                            cov: chex.Array) -> float:
  """Calculates the Gaussian log likelihood of a multivariate normal."""
  first_term = len(err) * jnp.log(2 * jnp.pi)
  _, second_term = jnp.linalg.slogdet(cov)
  third_term = jnp.einsum('ai,ab,bi->i', err, jnp.linalg.pinv(cov), err)
  return -0.5 * (first_term + second_term + third_term)  # pytype: disable=bad-return-type  # jax-types


def optimized_gaussian_ll(err: chex.Array) -> float:
  """Computes the Gaussian LL based on optimized residual MSE."""
  optimized_cov = jnp.mean(err ** 2) * jnp.eye(len(err))
  return gaussian_log_likelihood(err, optimized_cov)


@dataclasses.dataclass
class GaussianSampleKL(likelihood_base.SampleBasedKL):
  """Evaluates KL according to optimized Gaussian residual model."""
  num_test_seeds: int
  num_enn_samples: int
  enn_sigma: float
  key: chex.PRNGKey

  def __call__(
      self,
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: likelihood_base.GenerativeDataSampler,
  ) -> testbed_base.ENNQuality:
    """Evaluates KL according to optimized Gaussian residual model."""
    batched_sampler = jax.vmap(enn_sampler, in_axes=[None, 0])
    batched_ll = jax.vmap(gaussian_log_likelihood, in_axes=[0, None])

    def kl_estimate(key: chex.PRNGKey) -> float:
      """Computes KL estimate on a single instance of test data."""
      data, true_ll = data_sampler.test_data(key)
      tau = data.x.shape[0]
      data_keys = jax.random.split(key, self.num_enn_samples)
      samples = batched_sampler(data.x, data_keys)
      batched_err = samples - jnp.expand_dims(data.y, 0)
      chex.assert_shape(batched_err, [self.num_enn_samples, tau, 1])

      # ENN uses the enn_sigma to compute likelihood of sampled data
      enn_cov = self.enn_sigma ** 2 * jnp.eye(tau)
      sampled_ll = batched_ll(batched_err, enn_cov)
      chex.assert_shape(sampled_ll, [self.num_enn_samples, 1])

      # TODO(author2): Make sure of our KL computation.()
      ave_ll = metrics_base.average_sampled_log_likelihood(sampled_ll)  # pytype: disable=wrong-arg-types  # numpy-scalars
      return true_ll - ave_ll

    batched_kl = jax.jit(jax.vmap(kl_estimate))
    kl_keys = jax.random.split(self.key, self.num_test_seeds)
    sampled_kl = batched_kl(kl_keys)
    return testbed_base.ENNQuality(kl_estimate=jnp.mean(sampled_kl))


@dataclasses.dataclass
class GaussianSmoothedSampleKL(likelihood_base.SampleBasedKL):
  """Evaluates KL according to optimized Gaussian residual model."""
  num_test_seeds: int
  num_enn_samples: int
  enn_sigma: float
  key: chex.PRNGKey
  cov_ridge: float = 1e-6  # To smooth out the covariance estimate

  def __call__(
      self,
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: likelihood_base.GenerativeDataSampler,
  ) -> testbed_base.ENNQuality:
    """Evaluates KL according to optimized Gaussian residual model."""
    batched_sampler = jax.vmap(enn_sampler, in_axes=[None, 0])

    def kl_estimate(key: chex.PRNGKey) -> float:
      """Computes KL estimate on a single instance of test data."""
      data_key, enn_key = jax.random.split(key)
      data, true_ll = data_sampler.test_data(data_key)
      tau = data.x.shape[0]

      # Forward the ENN at many samples and form smoothed Gaussian approximation
      enn_keys = jax.random.split(enn_key, self.num_enn_samples)
      enn_samples = batched_sampler(data.x, enn_keys)
      enn_mean = jnp.mean(enn_samples, axis=0)
      chex.assert_shape(enn_mean, [tau, 1])

      # Estimates the covariance matrix with bias (simple variance in tau=1).
      enn_cov = jnp.cov(enn_samples[:, :, 0], rowvar=False, bias=True)
      if tau == 1:
        enn_cov = enn_cov[None, None]
      enn_cov += self.cov_ridge * jnp.eye(tau)
      chex.assert_shape(enn_cov, [tau, tau])

      # Estimate KL based on combined distribution
      err = data.y - enn_mean
      cov = enn_cov + self.enn_sigma ** 2 * jnp.eye(tau)
      unnormalized_kl = true_ll - gaussian_log_likelihood(err, cov)
      return unnormalized_kl

    batched_kl = jax.jit(jax.vmap(kl_estimate))
    kl_keys = jax.random.split(self.key, self.num_test_seeds)
    sampled_kl = batched_kl(kl_keys)
    return testbed_base.ENNQuality(kl_estimate=jnp.mean(sampled_kl))
