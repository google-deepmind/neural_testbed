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

"""CSV based logging system."""

import os
from typing import Any, Mapping

from neural_testbed import base as testbed_base
from neural_testbed.leaderboard import sweep
from neural_testbed.logging import base as logging_base
import pandas as pd

SAFE_SEPARATOR = '-'
INITIAL_SEPARATOR = '_-_'
GP_PREFIX = 'problem_id' + INITIAL_SEPARATOR


def wrap_problem(problem: testbed_base.TestbedProblem,
                 problem_id: str,
                 results_dir: str,
                 overwrite: bool = False) -> testbed_base.TestbedProblem:
  logger = Logger(problem_id, results_dir, overwrite)
  return logging_base.LoggingWrapper(problem, logger)


class Logger(logging_base.Logger):
  """Saves data to a CSV file via Pandas.

  In this simplified logger, each problem_id logs to a unique CSV index by
  problem_id. These are saved to a single results_dir by experiment.
  We strongly suggest that you use a *fresh* folder for each testbed run.

  This logger, along with the corresponding load functionality, serves as a
  simple, minimal example for users who need to implement logging to a different
  storage system.
  """

  def __init__(self,
               problem_id: str,
               results_dir: str = '/tmp/neural_testbed',
               overwrite: bool = False):
    """Initializes a new CSV logger."""

    if not os.path.exists(results_dir):
      try:
        os.makedirs(results_dir)
      except OSError:  # concurrent processes can makedir at same time
        pass

    # The default '/' symbol is dangerous for file systems!
    safe_problem_id = problem_id.replace(sweep.SEPARATOR, SAFE_SEPARATOR)
    filename = f'{GP_PREFIX}{safe_problem_id}.csv'
    save_path = os.path.join(results_dir, filename)

    if os.path.exists(save_path) and not overwrite:
      raise ValueError(
          f'File {save_path} already exists. Specify a different '
          'directory, or set overwrite=True to overwrite existing data.')

    self._data = []
    self._save_path = save_path

  def write(self, data: Mapping[str, Any]):
    """Adds a row to the internal list of data and saves to CSV."""
    self._data.append(data)
    df = pd.DataFrame(self._data)
    df.to_csv(self._save_path, index=False)
