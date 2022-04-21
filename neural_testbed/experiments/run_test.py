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
"""Tests for neural_testbed.experiments.run."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from neural_testbed import leaderboard
from neural_testbed.experiments import run

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class RunTest(parameterized.TestCase):

  @parameterized.parameters([[x] for x in leaderboard.CLASSIFICATION_2D_TEST])
  def test_neural_testbed(self, problem_id: str):
    FLAGS.problem_id = problem_id
    FLAGS.num_batches = 2
    run.main(None)


if __name__ == '__main__':
  absltest.main()
