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
"""Sweeps for Bayes by Backprop agent."""

from typing import Sequence

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import bbb


def base_sweep() -> Sequence[bbb.BBBConfig]:
  """Generates the bbb sweep over network parameters."""
  sweep = []
  for learning_rate in [1e-3, 3e-3]:
    for sigma_1 in [0.3, 0.5, 0.7, 1, 2]:
      for num_batches in [1000, 2000]:
        sweep.append(
            bbb.BBBConfig(
                learning_rate=learning_rate,
                sigma_1=sigma_1,
                num_batches=num_batches))
  return tuple(sweep)


def prior_sweep() -> Sequence[bbb.BBBConfig]:
  """Generates the bbb sweep over prior parameters."""
  sweep = []
  for sigma_1 in [1, 2, 4]:
    for sigma_2 in [0.3, 0.5, 0.7]:
      for mixture_scale in [0.5]:
        sweep.append(
            bbb.BBBConfig(
                sigma_1=sigma_1, sigma_2=sigma_2, mixture_scale=mixture_scale))
  return tuple(sweep)


def network_sweep() -> Sequence[bbb.BBBConfig]:
  """Generates the bbb sweep over network architecture for paper."""
  sweep = []
  for hidden_sizes in [(50, 50), (100, 100), (50, 50, 50)]:
    sweep.append(bbb.BBBConfig(hidden_sizes=hidden_sizes))
  return tuple(sweep)


def combined_sweep() -> Sequence[bbb.BBBConfig]:
  return tuple(base_sweep()) + tuple(prior_sweep()) + tuple(network_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=bbb.BBBConfig(),
      ctor=bbb.make_agent,
      sweep=combined_sweep,
  )
