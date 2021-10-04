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

"""Tests for neural_testbed.generative.classification_envlikelihood."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
from neural_testbed.generative.classification_envlikelihood import ClassificationEnvLikelihood
import numpy as np


class MLPClassificationEnsembleTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product([3, 10], [1, 3], [1, 3]))
  def test_valid_data(self, num_train: int, input_dim: int, tau: int):
    np.random.seed(0)
    num_class = 2
    rng = hk.PRNGSequence(0)

    x_generator = lambda k, n: jax.random.normal(k, [n, input_dim])
    fn_transformed = hk.without_apply_rng(hk.transform(
        lambda x: hk.nets.MLP([10, 10, num_class])(x)))  # pylint: disable=[unnecessary-lambda]
    params = fn_transformed.init(next(rng), np.zeros(shape=(input_dim,)))
    logit_fn = lambda x: fn_transformed.apply(params, x)

    mlp_model = ClassificationEnvLikelihood(
        logit_fn=logit_fn,
        x_generator=x_generator,
        num_train=num_train,
        key=next(rng),
        tau=tau,
    )

    # Check that the training data is reasonable.
    train_data = mlp_model.train_data
    assert train_data.x.shape == (num_train, input_dim)
    assert train_data.y.shape == (num_train, 1)
    assert np.all(~np.isnan(train_data.x))
    assert np.all(~np.isnan(train_data.y))

    # Check that the testing data is reasonable.
    for _ in range(3):
      test_data, log_likelihood = mlp_model.test_data(next(rng))
      assert np.isfinite(log_likelihood)
      assert test_data.x.shape == (tau, input_dim)
      assert test_data.y.shape == (tau, 1)
      assert np.all(~np.isnan(test_data.x))
      assert np.all(~np.isnan(test_data.y))

  @parameterized.parameters(itertools.product([1, 10, 100]))
  def test_not_all_test_data_same_x(self, num_train: int):
    """Generates testing data and checks not all the same x value."""
    np.random.seed(0)
    num_test_seeds = 10
    input_dim = 2
    num_class = 2
    tau = 1
    rng = hk.PRNGSequence(0)

    x_generator = lambda k, n: jax.random.normal(k, [n, input_dim])
    fn_transformed = hk.without_apply_rng(hk.transform(
        lambda x: hk.nets.MLP([10, 10, num_class])(x)))  # pylint: disable=[unnecessary-lambda]
    params = fn_transformed.init(next(rng), np.zeros(shape=(input_dim,)))
    logit_fn = lambda x: fn_transformed.apply(params, x)
    mlp_model = ClassificationEnvLikelihood(
        logit_fn=logit_fn,
        x_generator=x_generator,
        num_train=num_train,
        key=next(rng),
        tau=tau,
    )

    num_distinct_x = 0
    reference_data, _ = mlp_model.test_data(key=next(rng))
    for _ in range(num_test_seeds):
      test_data, _ = mlp_model.test_data(key=next(rng))
      if not np.all(np.isclose(test_data.x, reference_data.x)):
        num_distinct_x += 1
    assert num_distinct_x > 0

  @parameterized.parameters(itertools.product([10], [1], [10]))
  def test_valid_labels(self, num_train: int, input_dim: int, num_seeds: int):
    """Checks that for at most 20% of problems, the labels are degenerate."""
    num_class = 2
    tau = 1
    rng = hk.PRNGSequence(0)

    x_generator = lambda k, n: jax.random.normal(k, [n, input_dim])
    fn_transformed = hk.without_apply_rng(hk.transform(
        lambda x: hk.nets.MLP([10, 10, num_class])(x)))  # pylint: disable=[unnecessary-lambda]

    labels_means = []
    for _ in range(num_seeds):
      params = fn_transformed.init(next(rng), np.zeros(shape=(input_dim,)))
      logit_fn = functools.partial(fn_transformed.apply, params)
      mlp_model = ClassificationEnvLikelihood(
          logit_fn=logit_fn,
          x_generator=x_generator,
          num_train=num_train,
          key=next(rng),
          tau=tau,
      )

      train_data = mlp_model.train_data
      labels_means.append(np.mean(train_data.y.copy()))

    degenerate_cases = labels_means.count(0.) + labels_means.count(1.)
    # Check that for at most 20% of problems, the labels are degenerate
    assert degenerate_cases / num_seeds <= 0.2


if __name__ == '__main__':
  absltest.main()
