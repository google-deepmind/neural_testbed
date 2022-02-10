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
"""Sweeps for ensemble agent."""

from typing import Sequence

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import ensemble


def vanilla_sweep() -> Sequence[ensemble.VanillaEnsembleConfig]:
  sweep = []
  for num_ensemble in [1, 3, 10, 30, 100]:
    sweep.append(ensemble.VanillaEnsembleConfig(num_ensemble))
  return tuple(sweep)


def weight_sweep() -> Sequence[ensemble.VanillaEnsembleConfig]:
  sweep = []
  for adaptive_weight_scale in [True, False]:
    for l2_weight_decay in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
      sweep.append(ensemble.VanillaEnsembleConfig(
          num_ensemble=30,
          l2_weight_decay=l2_weight_decay,
          adaptive_weight_scale=adaptive_weight_scale,
      ))
  return tuple(sweep)


def combined_sweep() -> Sequence[ensemble.VanillaEnsembleConfig]:
  return tuple(vanilla_sweep()) + tuple(weight_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=ensemble.VanillaEnsembleConfig(),
      ctor=ensemble.make_agent,
      sweep=combined_sweep,
  )
