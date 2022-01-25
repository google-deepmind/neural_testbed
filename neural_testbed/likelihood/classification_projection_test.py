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

"""Tests for classification_projection."""

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from neural_testbed.likelihood import classification_projection
from neural_testbed.likelihood import utils


class _ProblemConfig(NamedTuple):
  probits: chex.Array
  y: chex.Array
  brute_ll: chex.Array


def make_naive_ensemble_problem(num_enn_samples: int,
                                tau: int,
                                num_classes: int,
                                key: chex.PRNGKey,
                                temperature: float = 1.) -> _ProblemConfig:
  """Create a naive ensemble problem."""
  logit_key, y_key = jax.random.split(key)

  # sample logits, probs, and probits
  logits = jax.random.normal(
      logit_key, shape=[num_enn_samples, tau, num_classes]) / temperature

  probs = jax.nn.softmax(logits)
  probits = jsp.ndtri(probs)
  chex.assert_shape(probits, [num_enn_samples, tau, num_classes])

  # sample random labels
  y = jax.random.categorical(y_key, logits[0, :, :])
  chex.assert_shape(y, [tau])

  # compute the log-likelihood in a brute-force way
  log_probs_correct = jnp.log(probs[:, jnp.arange(tau), y])
  chex.assert_shape(log_probs_correct, [num_enn_samples, tau])

  log_probs_model = jnp.sum(log_probs_correct, axis=1)
  chex.assert_shape(log_probs_model, [num_enn_samples])

  brute_ll = jsp.logsumexp(log_probs_model) - jnp.log(num_enn_samples)

  return _ProblemConfig(probits, y[:, None], brute_ll)


def make_repeated_ensemble_problem(num_enn_samples: int,
                                   tau: int,
                                   num_classes: int,
                                   key: chex.PRNGKey,
                                   temperature: float = 1.) -> _ProblemConfig:
  """Create a naive ensemble problem."""
  logit_key, y_key = jax.random.split(key)

  assert num_enn_samples % 2 == 0

  # sample logits, probs, and probits
  logits = jax.random.normal(
      logit_key, shape=[int(num_enn_samples / 2), tau, num_classes
                       ]) / temperature

  logits = jnp.concatenate([logits, logits], axis=0)

  probs = jax.nn.softmax(logits)
  probits = jsp.ndtri(probs)
  chex.assert_shape(probits, [num_enn_samples, tau, num_classes])

  # sample random labels
  y = jax.random.categorical(y_key, logits[0, :, :])
  chex.assert_shape(y, [tau])

  # compute the log-likelihood in a brute-force way
  log_probs_correct = jnp.log(probs[:, jnp.arange(tau), y])
  chex.assert_shape(log_probs_correct, [num_enn_samples, tau])

  log_probs_model = jnp.sum(log_probs_correct, axis=1)
  chex.assert_shape(log_probs_model, [num_enn_samples])

  brute_ll = jsp.logsumexp(log_probs_model) - jnp.log(num_enn_samples)

  return _ProblemConfig(probits, y[:, None], brute_ll)


def make_ll_estimate(cluster_alg: classification_projection.ClusterAlg):
  def ll_estimate(
      probits: chex.Array, y: chex.Array, key: chex.PRNGKey) -> chex.Array:
    # Perform appropriate clustering
    num_enn_samples = probits.shape[0]
    counts, centers = cluster_alg(probits, y, key)

    # Compute the model log likelihood
    model_ll = jax.jit(utils.log_sum_prod)(counts / num_enn_samples, centers)
    return model_ll
  return jax.jit(ll_estimate)


class ClassificationParityTest(parameterized.TestCase):

  @parameterized.product(
      num_enn_samples=[1000, 10000],
      tau=[10, 30],
      num_classes=[2, 4, 10],
      cluster_only_correct_class=[True, False],
  )
  def test_random_projection(
      self, num_enn_samples: int, tau: int, num_classes: int,
      cluster_only_correct_class: bool):
    rng = hk.PRNGSequence(999)
    probits, y, brute_ll = make_naive_ensemble_problem(
        num_enn_samples, tau, num_classes, key=next(rng))

    cluster_alg = classification_projection.RandomProjection(
        7, cluster_only_correct_class)
    ll_estimate = make_ll_estimate(cluster_alg)
    model_ll = ll_estimate(probits, y, next(rng))

    # Check the absolute error is not too high
    rel_err = float(jnp.abs(model_ll - brute_ll)) / float(jnp.abs(brute_ll))
    assert rel_err < 0.25, f'relative error is {rel_err}'

  @parameterized.product(
      num_enn_samples=[1000, 10000],
      tau=[10, 30],
      num_classes=[2, 4, 10],
      cluster_only_correct_class=[True, False],
  )
  def test_random_projection_repeated(
      self, num_enn_samples: int, tau: int, num_classes: int,
      cluster_only_correct_class: bool):
    rng = hk.PRNGSequence(999)
    probits, y, brute_ll = make_repeated_ensemble_problem(
        num_enn_samples, tau, num_classes, key=next(rng))

    cluster_alg = classification_projection.RandomProjection(
        7, cluster_only_correct_class)
    ll_estimate = make_ll_estimate(cluster_alg)
    model_ll = ll_estimate(probits, y, next(rng))

    # Check the absolute error is not too high
    rel_err = float(jnp.abs(model_ll - brute_ll)) / float(jnp.abs(brute_ll))
    assert rel_err < 0.30, f'relative error is {rel_err}'

  @parameterized.product(
      num_enn_samples=[100, 1000],
      tau=[10, 30],
      num_classes=[2, 4, 10],
      cluster_only_correct_class=[True, False],
  )
  def test_kmeans_cluster(
      self, num_enn_samples: int, tau: int, num_classes: int,
      cluster_only_correct_class: bool):
    rng = hk.PRNGSequence(999)
    probits, y, brute_ll = make_naive_ensemble_problem(
        num_enn_samples, tau, num_classes, key=next(rng))

    cluster_alg = classification_projection.KmeansCluster(
        num_enn_samples, max_iter=10,
        cluster_only_correct_class=cluster_only_correct_class)
    ll_estimate = make_ll_estimate(cluster_alg)
    model_ll = ll_estimate(probits, y, next(rng))

    # TODO(author2): Push this relative error factor lower
    # Check the absolute error is not too high
    rel_err = float(jnp.abs(model_ll - brute_ll)) / float(jnp.abs(brute_ll))
    assert rel_err < 0.01, f'relative error is {rel_err}'


if __name__ == '__main__':
  absltest.main()
