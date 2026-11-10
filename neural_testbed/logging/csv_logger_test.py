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
import shutil
import stat

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from neural_testbed.logging import csv_logger
import pandas as pd

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class CsvLoggerTest(parameterized.TestCase):

  def test_default_results_dir_creates_private_temp_dir_and_writes_csv(self):
    logger = csv_logger.Logger('classification_2d/0')
    self.addCleanup(shutil.rmtree, logger.results_dir, ignore_errors=True)

    self.assertTrue(os.path.isdir(logger.results_dir))
    self.assertStartsWith(
        os.path.basename(logger.results_dir), 'neural_testbed_'
    )
    if os.name == 'posix':
      mode = stat.S_IMODE(os.lstat(logger.results_dir).st_mode)
      self.assertEqual(mode, 0o700)

    logger.write({'kl_estimate': 1.25})
    df = pd.read_csv(logger.save_path)
    self.assertSequenceEqual(list(df['kl_estimate']), [1.25])

  def test_write_refuses_symlink_save_path_without_modifying_target(self):
    results_dir = self.create_tempdir().full_path
    victim_file = self.create_tempfile(content='ORIGINAL_CONTENT').full_path
    symlink_path = os.path.join(
        results_dir, 'problem_id_-_classification_2d-0.csv'
    )
    os.symlink(victim_file, symlink_path)

    logger = csv_logger.Logger(
        'classification_2d/0', results_dir=results_dir, overwrite=True
    )
    with self.assertRaisesRegex(RuntimeError, 'symbolic link'):
      logger.write({'kl_estimate': 1.0})

    with open(victim_file, 'r', encoding='utf-8') as f:
      self.assertEqual(f.read(), 'ORIGINAL_CONTENT')

  def test_write_refuses_symlink_planted_after_init(self):
    results_dir = self.create_tempdir().full_path
    victim_file = self.create_tempfile(content='ORIGINAL_CONTENT').full_path

    logger = csv_logger.Logger(
        'classification_2d/0', results_dir=results_dir, overwrite=True
    )
    os.symlink(victim_file, logger.save_path)

    with self.assertRaisesRegex(RuntimeError, 'symbolic link'):
      logger.write({'kl_estimate': 1.0})

    with open(victim_file, 'r', encoding='utf-8') as f:
      self.assertEqual(f.read(), 'ORIGINAL_CONTENT')

  def test_init_refuses_dangling_symlink_when_overwrite_false(self):
    results_dir = self.create_tempdir().full_path
    missing_target = os.path.join(results_dir, 'missing_target.txt')
    symlink_path = os.path.join(
        results_dir, 'problem_id_-_classification_2d-0.csv'
    )
    os.symlink(missing_target, symlink_path)

    with self.assertRaisesRegex(ValueError, 'already exists'):
      csv_logger.Logger(
          'classification_2d/0', results_dir=results_dir, overwrite=False
      )

  @parameterized.named_parameters(
      dict(testcase_name='bare_path', suffix=''),
      dict(testcase_name='trailing_sep', suffix=os.sep),
  )
  def test_init_refuses_symlink_results_dir(self, suffix: str):
    parent_dir = self.create_tempdir().full_path
    target_dir = self.create_tempdir().full_path
    symlink_dir = os.path.join(parent_dir, 'linked_results')
    os.symlink(target_dir, symlink_dir)

    with self.assertRaisesRegex(RuntimeError, 'symbolic link'):
      csv_logger.Logger(
          'classification_2d/0',
          results_dir=symlink_dir + suffix,
          overwrite=True,
      )


if __name__ == '__main__':
  absltest.main()
