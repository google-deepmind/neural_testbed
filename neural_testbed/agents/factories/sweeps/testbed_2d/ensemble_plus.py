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
"""Sweeps for ensemble_plus agent."""

from typing import Sequence

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import ensemble_plus


def basic_sweep() -> Sequence[ensemble_plus.EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for num_ensemble in [1, 3, 10, 30, 100]:
    sweep.append(ensemble_plus.EnsembleConfig(
        num_ensemble=num_ensemble,
    ))
  return tuple(sweep)


def boot_sweep() -> Sequence[ensemble_plus.EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for distribution in ['none', 'bernoulli', 'exponential']:
    sweep.append(ensemble_plus.EnsembleConfig(
        distribution=distribution,
    ))
  return tuple(sweep)


def weight_decay_sweep() -> Sequence[ensemble_plus.EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for l2_weight_decay in [0.1, 0.3, 1, 3, 10]:
    sweep.append(ensemble_plus.EnsembleConfig(
        l2_weight_decay=l2_weight_decay,
    ))
  return tuple(sweep)


def combined_sweep() -> Sequence[ensemble_plus.EnsembleConfig]:
  return tuple(basic_sweep()) + tuple(boot_sweep()) + tuple(
      weight_decay_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=ensemble_plus.EnsembleConfig(),
      ctor=ensemble_plus.make_agent,
      sweep=combined_sweep,
  )
