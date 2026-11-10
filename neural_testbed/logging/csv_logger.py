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

import errno
import os
import tempfile
from typing import Any, Mapping, Optional

from neural_testbed import base as testbed_base
from neural_testbed.leaderboard import sweep
from neural_testbed.logging import base as logging_base
import pandas as pd

SAFE_SEPARATOR = '-'
INITIAL_SEPARATOR = '_-_'
GP_PREFIX = 'problem_id' + INITIAL_SEPARATOR


def _is_symlink_dir(path: str) -> bool:
  stripped = path.rstrip(os.sep)
  if os.altsep:
    stripped = stripped.rstrip(os.altsep)
  return os.path.islink(stripped or os.sep)


def wrap_problem(
    problem: testbed_base.TestbedProblem,
    problem_id: str,
    results_dir: Optional[str] = None,
    overwrite: bool = False,
) -> testbed_base.TestbedProblem:
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

  def __init__(
      self,
      problem_id: str,
      results_dir: Optional[str] = None,
      overwrite: bool = False,
  ):
    """Initializes a new CSV logger."""
    if results_dir is None:
      results_dir = tempfile.mkdtemp(prefix='neural_testbed_')
    else:
      if _is_symlink_dir(results_dir):
        raise RuntimeError(
            f'Refusing to use {results_dir}: results_dir is a symbolic link.'
        )
      if not os.path.exists(results_dir):
        try:
          os.makedirs(results_dir, mode=0o700, exist_ok=True)
        except OSError:  # concurrent processes can makedir at same time
          if not os.path.isdir(results_dir):
            raise
      if _is_symlink_dir(results_dir):
        raise RuntimeError(
            f'Refusing to use {results_dir}: results_dir is a symbolic link.'
        )

    # The default '/' symbol is dangerous for file systems!
    safe_problem_id = problem_id.replace(sweep.SEPARATOR, SAFE_SEPARATOR)
    safe_problem_id = safe_problem_id.replace(os.sep, SAFE_SEPARATOR)
    if os.altsep:
      safe_problem_id = safe_problem_id.replace(os.altsep, SAFE_SEPARATOR)
    filename = f'{GP_PREFIX}{safe_problem_id}.csv'
    save_path = os.path.join(results_dir, filename)

    if os.path.lexists(save_path) and not overwrite:
      raise ValueError(
          f'File {save_path} already exists. Specify a different '
          'directory, or set overwrite=True to overwrite existing data.'
      )

    self._data = []
    self._results_dir = results_dir
    self._save_path = save_path

  @property
  def results_dir(self) -> str:
    return self._results_dir

  @property
  def save_path(self) -> str:
    return self._save_path

  def write(self, data: Mapping[str, Any]):
    """Adds a row to the internal list of data and saves to CSV."""
    self._data.append(data)
    df = pd.DataFrame(self._data)
    if _is_symlink_dir(self._results_dir):
      raise RuntimeError(
          f'Refusing to write to {self._save_path}: '
          f'{self._results_dir} is a symbolic link.'
      )
    if os.path.islink(self._save_path):
      raise RuntimeError(
          f'Refusing to write to {self._save_path}: path is a symbolic link.'
      )
    open_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    )
    try:
      fd = os.open(self._save_path, open_flags, 0o600)
    except OSError as e:
      if e.errno == errno.ELOOP:
        raise RuntimeError(
            f'Refusing to write to {self._save_path}: path is a symbolic link.'
        ) from e
      raise
    try:
      f = os.fdopen(fd, 'w', encoding='utf-8', newline='')
    except Exception:
      os.close(fd)
      raise
    with f:
      df.to_csv(f, index=False)
