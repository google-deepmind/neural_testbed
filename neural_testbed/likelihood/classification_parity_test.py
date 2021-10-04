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

"""Tests for neural_testbed.likelihood.classification_parity."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed.likelihood import classification_parity


def make_low_dimensional_problem(
    input_dim: int,
    latent_dim: int,
    num_samples: int,
    prob: float,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
  """Generate y_target and y_samples for evaluation."""
  rng = hk.PRNGSequence(key)
  matrix = jax.random.bernoulli(next(rng), p=0.5, shape=[latent_dim, input_dim])
  z_bar = jax.random.bernoulli(next(rng), p=prob, shape=[latent_dim])
  z = jax.random.bernoulli(next(rng), p=prob, shape=[num_samples, latent_dim])

  y_target = jnp.dot(z_bar, matrix).astype(jnp.int16) % 2
  y_samples = jnp.dot(z, matrix).astype(jnp.int16) % 2
  return y_target, y_samples


def brute_force_estimator(y_target: chex.Array, y_samples: chex.Array) -> float:
  batched_equal = jax.vmap(jnp.array_equal, in_axes=(0, None))
  return jnp.mean(batched_equal(y_samples, y_target))


class ClassificationParityTest(parameterized.TestCase):

  @parameterized.product(
      input_dim=[10, 100, 1000],
      latent_dim=[2, 4],
      num_features=[8],
  )
  def test_low_dim_recovery_parity_estimator(
      self, input_dim: int, latent_dim: int, num_features: int):
    seed = 0
    rng = hk.PRNGSequence(seed)
    y_target, y_samples = make_low_dimensional_problem(
        input_dim, latent_dim, num_samples=1000, prob=0.5, key=next(rng))
    parity_estimator = classification_parity.make_parity_check_estimator(
        num_features, next(rng))
    parity_prob = jnp.exp(parity_estimator(y_target, y_samples))
    brute_prob = brute_force_estimator(y_target, y_samples)

    # Check the absolute error is not too high
    abs_err = float(jnp.abs(parity_prob - brute_prob))
    assert abs_err < 0.05, f'absolute error is {abs_err}'

    # Check the relative error is not too high
    rel_err = abs_err / float(brute_prob)
    assert rel_err < 0.1, f'relative error is {rel_err}'


if __name__ == '__main__':
  absltest.main()
