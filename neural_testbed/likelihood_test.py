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

"""Tests for neural_testbed.likelihood."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
import dataclasses
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed import likelihood


class BernoulliDataSampler(likelihood.GenerativeDataSampler):
  """Generates data sampled from a fixed Bernoulli(p)."""

  def __init__(self, prob: float):
    self.probs = jnp.array([1 - prob, prob])
    self.x = jnp.ones([1, 1])

  @property
  def train_data(self) -> testbed_base.Data:
    raise ValueError('This problem should not be used for training.')

  def test_data(self, seed: int) -> Tuple[testbed_base.Data, float]:
    """Generates a random sample of test data with posterior log-likelihood."""
    bool_sample = jax.random.bernoulli(jax.random.PRNGKey(seed), self.probs[1])
    y = jnp.expand_dims(jnp.array(bool_sample, dtype=jnp.int32), 0)
    log_likelihood = jnp.log(self.probs[y])
    y = y[:, None]
    chex.assert_shape(y, [1, 1])
    return testbed_base.Data(self.x, y), log_likelihood


@dataclasses.dataclass
class BernoulliEpistemicSampler(testbed_base.EpistemicSampler):
  """ENN samples [0, logit_scale] with prob=p and [logit_scale, 0] with prob=1-p."""
  prob: float = 0.5
  logit_scale: float = 1e6

  def __call__(self, x: chex.Array, seed: int = 0) -> chex.Array:
    num_data = x.shape[0]
    bool_sample = jax.random.bernoulli(
        jax.random.PRNGKey(seed), self.prob, shape=[num_data])
    y = jnp.array(bool_sample, dtype=jnp.int32)
    logits = jax.nn.one_hot(y, num_classes=2) * self.logit_scale
    chex.assert_shape(logits, [num_data, 2])
    return logits


class UtilTest(parameterized.TestCase):

  @parameterized.parameters([1, 3, 100])
  def test_average_sampled_log_likelihood_all_neginf(self, num_sample: int):
    """Test that average of negative infinity log likelihood is neg infinity."""
    log_likelihood = jnp.concatenate([jnp.array([-jnp.inf] * num_sample)])
    avg_log_likelihood = likelihood.average_sampled_log_likelihood(
        log_likelihood)
    self.assertTrue(jnp.isneginf(avg_log_likelihood))

  @parameterized.parameters([3, 100])
  def test_average_sampled_log_likelihood_single_neginf(self, num_sample: int):
    """Test that avg with one negative infinity log likelihood is correct."""
    log_likelihood = jnp.concatenate([jnp.array([-jnp.inf]),
                                      jnp.zeros(shape=(num_sample - 1,))])
    avg_log_likelihood = likelihood.average_sampled_log_likelihood(
        log_likelihood)
    expected_log_likelihood = jnp.log((num_sample -1) / num_sample)
    self.assertAlmostEqual(
        avg_log_likelihood, expected_log_likelihood,
        msg=(f'Expected log likelihood to be {expected_log_likelihood} ',
             f'but received {avg_log_likelihood}'),
        delta=0.1/num_sample)

  @parameterized.product(
      ll_val=[-1000, -100, 10, 0],
      num_sample=[1, 3, 100])
  def test_average_sampled_log_likelihood_const_values(
      self, ll_val: float, num_sample: int):
    """Test that average of equal log likelihood values is correct."""
    log_likelihood = ll_val * jnp.ones(shape=(num_sample,))
    avg_log_likelihood = likelihood.average_sampled_log_likelihood(
        log_likelihood)
    self.assertAlmostEqual(
        avg_log_likelihood, ll_val,
        msg=(f'Expected log likelihood to be {ll_val} ',
             f'but received {avg_log_likelihood}'),
        delta=1e-5)

  @parameterized.product(
      err_val=[0, 1., 1e3],
      cov_val=[1, 1e-3, 1e3])
  def test_gaussian_log_likelihood_diagonal_cov(
      self, err_val: float, cov_val: float):
    """Test the computed log likelihood in the simple case of diagonal cov."""
    num_sample = 4
    err = err_val * jnp.ones(shape=(num_sample, 1))
    cov = cov_val * jnp.eye(num_sample)
    log_likelihood = likelihood.gaussian_log_likelihood(err, cov)
    expected_log_likelihood = -0.5 * (
        num_sample * jnp.log(2 * jnp.pi * cov_val)
        + (jnp.linalg.norm(err) ** 2) / cov_val)
    self.assertAlmostEqual(
        log_likelihood, expected_log_likelihood,
        msg=(f'Expected log likelihood to be {expected_log_likelihood} ',
             f'but received {log_likelihood}'),
        delta=1e-5)

  @parameterized.product(
      true_prob=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
      enn_err=[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2])
  def test_bernoulli_sample_based_kl(self, true_prob: float, enn_err: float):
    """Tests the estimated sample-based KL is close to the analytic KL.

    Compares the bernoulli sample-based estimate against an analytic KL.
    Checks that the absolute error is less than 0.01.

    Args:
      true_prob: true probability of class 1 in generative model.
      enn_err: error in enn probability estimate (clipped to 0, 1).
    """
    enn_prob = jnp.clip(true_prob + enn_err, 0, 1)
    true_kl = (true_prob * jnp.log(true_prob / enn_prob)
               + (1 - true_prob) * jnp.log((1- true_prob) / (1 - enn_prob)))
    kl_estimator = likelihood.CategoricalSampleKL(
        num_test_seeds=10_000, num_enn_samples=10_000)
    sample_kl = kl_estimator(
        BernoulliEpistemicSampler(enn_prob), BernoulliDataSampler(true_prob))
    self.assertAlmostEqual(
        true_kl, sample_kl.kl_estimate,
        msg=f'Expected KL={true_kl} but received {sample_kl}',
        delta=1e-2,
    )


if __name__ == '__main__':
  absltest.main()
