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
"""Sweep for Random Forest agent."""

from typing import Sequence

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import random_forest


def rf_sweep() -> Sequence[random_forest.RandomForestConfig]:
  sweep = []
  for n_estimators in [10, 100, 1000]:
    for criterion in ['gini', 'entropy']:
      sweep.append(random_forest.RandomForestConfig(n_estimators, criterion))
  return tuple(sweep)


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=random_forest.RandomForestConfig(),
      ctor=random_forest.make_agent,
      sweep=rf_sweep,
  )
