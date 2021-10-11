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
"""Tests for efficient_agent.neural_testbed.generative.plotting."""


from absl.testing import absltest
from absl.testing import parameterized

import chex
from enn import base as enn_base
import jax
import jax.numpy as jnp
from neural_testbed.generative import plotting
from neural_testbed.leaderboard import load
from neural_testbed.leaderboard import sweep


def regression_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
  mean = jax.random.normal(key)
  return jnp.ones([x.shape[0], 1]) * mean


def classification_sampler(x: enn_base.Array,
                           key: chex.PRNGKey) -> enn_base.Array:
  del key
  return jnp.zeros([x.shape[0], 2])


class PlottingTest(parameterized.TestCase):

  @parameterized.parameters([[x] for x in sweep.CLASSIFICATION_2D_TEST])
  def test_2d_classification(self, problem_id: str):
    """Check that the 1d classification plot doesn't fail."""
    problem = load.problem_from_id(problem_id)
    _ = plotting.sanity_plots(problem, classification_sampler)

  @parameterized.parameters([[x] for x in sweep.REGRESSION_TEST])
  def test_1d_regression(self, problem_id: str):
    """Check that the 1d regression plot doesn't fail."""
    problem = load.problem_from_id(problem_id)
    _ = plotting.sanity_plots(problem, regression_sampler)

  @parameterized.parameters([[x] for x in sweep.ENN_PAPER_TEST])
  def test_1d_enn_paper(self, problem_id: str):
    """Check that the 1d enn_paper plot doesn't fail."""
    problem = load.problem_from_id(problem_id)
    _ = plotting.sanity_plots(problem, regression_sampler)

if __name__ == '__main__':
  absltest.main()
