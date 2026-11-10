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

import os
import sys

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from neural_testbed.leaderboard import entries_csv
from neural_testbed.leaderboard import score
from neural_testbed.leaderboard import score_csv
from neural_testbed.leaderboard import sweep
from neural_testbed.logging import csv_logger

FLAGS = flags.FLAGS


def log_fake_results(problem_id: str, results_dir: str) -> None:
  """Populate a fake set of results."""
  logger = csv_logger.Logger(problem_id, results_dir)
  logger.write({
      'kl_estimate': 10.0,
      'total_seconds': 2.0,
      'train_seconds': 1.0,
  })


class ScoreCsvTest(parameterized.TestCase):

  @parameterized.parameters([['cool_agent'], ['uncool_agent']])
  def test_logger(self, name: str):
    """Write some fake results to csv and then load them back in."""
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    results_dir = self.create_tempdir().full_path

    for problem_id in sweep.CLASSIFICATION_2D[:10]:
      log_fake_results(problem_id=problem_id, results_dir=results_dir)

    # Make a fake entry with this given name, and load it back in.
    entry = entries_csv.Entry(name, results_dir)
    data = score_csv.load_entries(entry)

    # Check that the data is the right type
    self.assertIsInstance(
        data, score.LeaderboardData, 'Data is not the right type'
    )

    # Check that the agent name has been passed through
    self.assertIn(name, data.names, 'the agent name has been passed through.')

    # Check that sweep metadata is joined correctly on problem_id
    self.assertIn(
        'problem_id',
        data.df.columns,
        'sweep metadata is not joined correctly on problem_id.',
    )

    # Check that we only loaded one agent
    self.assertLen(data.agents, 1)
    agent_data = data.agents[0]
    self.assertIsInstance(
        agent_data, score.AgentData, 'Agent data is not the right type.'
    )

    # Check the quality of this single agent data
    self.assertEqual(
        agent_data.name, name, 'Agent data does not have the correct name.'
    )
    self.assertLess(agent_data.pct_health, 0.5, 'Health is less that 50%.')

  def test_load_entries_refuses_symlink_csv(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      flags.FLAGS(sys.argv)
    results_dir = self.create_tempdir().full_path
    target_file = self.create_tempfile(content='kl_estimate\n1.0\n').full_path
    symlink_csv = os.path.join(
        results_dir, 'problem_id_-_classification_2d-0.csv'
    )
    os.symlink(target_file, symlink_csv)

    entry = entries_csv.Entry('agent', results_dir)
    with self.assertRaisesRegex(RuntimeError, 'symbolic link'):
      score_csv.load_entries(entry)

  def test_load_entries_refuses_symlink_results_dir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      flags.FLAGS(sys.argv)
    parent_dir = self.create_tempdir().full_path
    target_dir = self.create_tempdir().full_path
    symlink_dir = os.path.join(parent_dir, 'linked_dir')
    os.symlink(target_dir, symlink_dir)

    entry = entries_csv.Entry('agent', symlink_dir + os.sep)
    with self.assertRaisesRegex(RuntimeError, 'symbolic link'):
      score_csv.load_entries(entry)

  def test_get_baselines_uses_explicit_results_dir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      flags.FLAGS(sys.argv)
    results_dir = self.create_tempdir().full_path
    baselines = entries_csv.get_baselines(results_dir)
    self.assertSequenceEqual(
        baselines, [entries_csv.Entry('uniform', results_dir)]
    )


if __name__ == '__main__':
  absltest.main()
