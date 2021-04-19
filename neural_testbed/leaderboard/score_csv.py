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

"""Load testbed entries from csv file."""

import glob
import os
from typing import Any, Sequence, Tuple

from neural_testbed.leaderboard import entries_csv
from neural_testbed.leaderboard import score
from neural_testbed.leaderboard import sweep
from neural_testbed.logging import csv_logger
import pandas as pd


def _load_entry(entry: entries_csv.Entry) -> Tuple[pd.DataFrame, Sequence[str]]:
  """Loads a single entry from csv logs."""
  data = []
  results_dir = entry.results_dir
  for file_path in glob.glob(os.path.join(results_dir, '*.csv')):
    _, name = os.path.split(file_path)
    # Rough and ready error-checking for only neural testbed csv files.
    if not name.startswith(csv_logger.GP_PREFIX):
      print('Warning - we recommend you use a fresh folder for bsuite results.')
      continue

    # Then we will assume that the file is actually a neural testbed result
    df = pd.read_csv(file_path)
    file_gp_id = name.strip('.csv').split(csv_logger.INITIAL_SEPARATOR)[1]
    gp_id = file_gp_id.replace(csv_logger.SAFE_SEPARATOR, sweep.SEPARATOR)
    df['gp_id'] = gp_id
    df['results_dir'] = results_dir
    data.append(df)
  df = pd.concat(data, sort=False)
  return df, []


def load_entries(
    leaderboard_entries: Any,
    leaderboard_sweep: Sequence[str] = sweep.CLASSIFICATION,
    verbose: bool = True,
) -> score.LeaderboardData:
  """Loads leaderboard entries and outputs a list of cleaned AgentData."""
  return score.LeaderboardData(*score.load_entries(
      leaderboard_entries, _load_entry, leaderboard_sweep, verbose))
