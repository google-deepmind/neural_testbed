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

"""Tests for neural_testbed.leaderboard.load."""

from absl.testing import absltest
from absl.testing import parameterized
from neural_testbed.leaderboard import load
import numpy as np


class LoadTest(parameterized.TestCase):

  @parameterized.parameters([
      ['classification_2d/0'],
      ['classification_2d/10'],
      ['classification_2d/100'],
  ])
  def test_gp_loading(self, problem_id: str):
    """Tests you can load from problem_id and data format matches prior."""
    testbed_problem = load.problem_from_id(problem_id)
    data = testbed_problem.train_data
    prior = testbed_problem.prior_knowledge
    assert data.x.shape == (prior.num_train, prior.input_dim)
    assert data.y.shape == (prior.num_train, 1)
    assert np.all(~np.isnan(data.x))
    assert np.all(~np.isnan(data.y))


if __name__ == '__main__':
  absltest.main()
