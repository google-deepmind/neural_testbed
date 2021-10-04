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

"""In-memory logging system."""

from typing import Any, Mapping

from neural_testbed import base as testbed_base
from neural_testbed.logging import base as logging_base
import pandas as pd


def wrap_problem(
    problem: testbed_base.TestbedProblem) -> testbed_base.TestbedProblem:
  return logging_base.LoggingWrapper(problem, Logger())


class Logger(logging_base.Logger):
  """Saves data to python memory."""

  def __init__(self):
    """Initializes a new python in-memory logger."""
    self._data = []

  def write(self, data: Mapping[str, Any]):
    """Adds a row to the internal list of data and saves to CSV."""
    self._data.append(data)

  @property
  def df(self) -> pd.DataFrame:
    return pd.DataFrame(self._data)
