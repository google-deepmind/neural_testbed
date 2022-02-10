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
"""Sweeps for deep kernel agent."""

from typing import Sequence

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import deep_kernel


def deep_kernel_sweep() -> Sequence[deep_kernel.DeepKernelConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for scale_factor in [1., 2., 3., 4., 5., 6.]:
    for sigma_squared_factor in [0.5, 1., 2., 3., 4.]:
      sweep.append(
          deep_kernel.DeepKernelConfig(
              scale_factor=scale_factor,
              sigma_squared_factor=sigma_squared_factor))
  return tuple(sweep)


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=deep_kernel.DeepKernelConfig(),
      ctor=deep_kernel.make_agent,
      sweep=deep_kernel_sweep,
  )
