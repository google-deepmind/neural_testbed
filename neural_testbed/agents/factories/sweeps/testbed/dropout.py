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
"""Sweeps for MC Dropout agent."""

from typing import Sequence

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import dropout


def droprate_sweep() -> Sequence[dropout.McDropoutConfig]:
  """Generates the dropout sweep over dropping parameters for paper."""
  sweep = []
  for dropout_rate in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for dropout_input in [True, False]:
      sweep.append(
          dropout.McDropoutConfig(
              dropout_rate=dropout_rate,
              dropout_input=dropout_input,
              batch_strategy=True,
              hidden_sizes=(50, 50)))
  return tuple(sweep)


def l2reg_sweep() -> Sequence[dropout.McDropoutConfig]:
  """Generates the dropout sweep over l2 regularization parameters for paper."""
  sweep = []
  for adaptive_weight_scale in [True, False]:
    for length_scale in [0.01, 0.1, 0.3, 1, 3, 10]:
      sweep.append(
          dropout.McDropoutConfig(
              adaptive_weight_scale=adaptive_weight_scale,
              length_scale=length_scale,
              hidden_sizes=(50, 50)))
  return tuple(sweep)


def network_sweep() -> Sequence[dropout.McDropoutConfig]:
  """Generates the dropout sweep over dropping parameters for paper."""
  sweep = []
  for hidden_sizes in [(50, 50), (100, 100)]:
    sweep.append(
        dropout.McDropoutConfig(
            hidden_sizes=hidden_sizes,
            batch_strategy=True,
        ))
  return tuple(sweep)


def batch_sweep() -> Sequence[dropout.McDropoutConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for batch_strategy in [True, False]:
    for num_batches in [500, 1000]:
      sweep.append(
          dropout.McDropoutConfig(
              batch_strategy=batch_strategy,
              num_batches=num_batches,
              hidden_sizes=(50, 50)))
  return tuple(sweep)


def combined_sweep() -> Sequence[dropout.McDropoutConfig]:
  return tuple(droprate_sweep()) + tuple(l2reg_sweep()) + tuple(
      network_sweep()) + tuple(batch_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=dropout.McDropoutConfig(),
      ctor=dropout.make_mc_dropout_agent,
      sweep=combined_sweep,
      )
