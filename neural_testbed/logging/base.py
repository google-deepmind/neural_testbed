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

"""An abstract base class for loggers."""

import abc
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from acme.utils import loggers
import chex
from neural_testbed import base as testbed_base
import pandas as pd

# TODO(author2): turn this into a typevar typed on the entry Type
Entry = Any
# Returns a dataframe with the results and a sequence of sweep_vars
EntryLoader = Callable[[Entry], Tuple[pd.DataFrame, Sequence[str]]]


class Logger(abc.ABC):
  """A logger has a `write` method."""

  @abc.abstractmethod
  def write(self, data: Mapping[str, Any]):
    """Writes `data` to destination (file, terminal, database, etc)."""


class LoggingWrapper(testbed_base.TestbedProblem):
  """Wraps a testbed problem with a logger."""

  def __init__(self,
               problem: testbed_base.TestbedProblem,
               logger: Logger):
    self._problem = problem
    self._logger = logger
    self._start = time.time()
    self._train_start = self._start

  @property
  def train_data(self) -> testbed_base.Data:
    self._train_start = time.time()
    return self._problem.train_data

  def evaluate_quality(
      self,
      enn_sampler: testbed_base.EpistemicSampler) -> testbed_base.ENNQuality:
    # Before evaluating enn, we record the time at the end of training.
    train_end = time.time()
    enn_quality = self._problem.evaluate_quality(enn_sampler)
    results = {
        'kl_estimate': float(enn_quality.kl_estimate),
        'total_seconds': time.time() - self._start,
        'train_seconds': train_end - self._train_start,
        'evaluation_seconds': time.time() - train_end,
    }
    if enn_quality.extra:
      extra_results = clean_results(enn_quality.extra)
      results.update({
          key: value for key, value in extra_results.items()
      })
    self._logger.write(results)
    return enn_quality

  @property
  def prior_knowledge(self) -> testbed_base.PriorKnowledge:
    return self._problem.prior_knowledge

  @property
  def problem(self) -> testbed_base.TestbedProblem:
    problem = self._problem
    if hasattr(problem, 'problem'):
      return problem.problem
    return problem


def clean_results(results: Dict[str, Any]) -> Dict[str, Any]:
  """Cleans the results for logging (can't log jax arrays)."""
  def clean_result(value: Any) -> Any:
    value = loggers.to_numpy(value)
    if isinstance(value, chex.ArrayNumpy) and value.size == 1:
      value = float(value)
    return value

  for key, value in results.items():
    results[key] = clean_result(value)

  return results
