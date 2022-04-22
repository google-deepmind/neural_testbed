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
"""Sweeps for hypermodel agent."""

from typing import Sequence

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import hypermodel


def l2reg_sweep() -> Sequence[hypermodel.HypermodelConfig]:
  """Generates the hypermodel sweep over l2 regularization parameters for paper."""
  sweep = []
  for l2_weight_decay in [0.1, 0.3, 1, 3, 10]:
    sweep.append(hypermodel.HypermodelConfig(l2_weight_decay=l2_weight_decay,
                                             num_batches=1000,
                                             batch_strategy=True,))
  return tuple(sweep)


def index_sweep() -> Sequence[hypermodel.HypermodelConfig]:
  """Generates the hypermodel sweep over basic parameters for paper."""
  sweep = []
  for index_dim in [1, 3, 5, 7]:
    sweep.append(
        hypermodel.HypermodelConfig(
            index_dim=index_dim,
            num_batches=1000,
            batch_strategy=True,
        ))
  return tuple(sweep)


def boot_sweep() -> Sequence[hypermodel.HypermodelConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for distribution in ['none', 'bernoulli', 'exponential']:
    sweep.append(
        hypermodel.HypermodelConfig(
            distribution=distribution,
            num_batches=1000,
            batch_strategy=True,
        ))
  return tuple(sweep)


def prior_sweep() -> Sequence[hypermodel.HypermodelConfig]:
  """Generates the hypermodel sweep over prior function parameters for paper."""
  sweep = []
  for prior_hidden_sizes in [(10,), (10, 10)]:
    for prior_scale in [1, 3]:
      for temp_scale_prior in ['lin', 'sqrt']:
        sweep.append(
            hypermodel.HypermodelConfig(
                prior_hidden_sizes=prior_hidden_sizes,
                prior_scale=prior_scale,
                temp_scale_prior=temp_scale_prior,
                num_batches=1000,
                batch_strategy=True,))
  return tuple(sweep)


def batch_sweep() -> Sequence[hypermodel.HypermodelConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for batch_strategy in [True, False]:
    for num_batches in [500, 1000]:
      sweep.append(
          hypermodel.HypermodelConfig(
              batch_strategy=batch_strategy,
              num_batches=num_batches))
  return tuple(sweep)


def combined_sweep() -> Sequence[hypermodel.HypermodelConfig]:
  return tuple(prior_sweep()) + tuple(index_sweep()) + tuple(
      l2reg_sweep()) + tuple(boot_sweep()) + tuple(batch_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=hypermodel.HypermodelConfig(),
      ctor=hypermodel.make_hypermodel_agent,
      sweep=combined_sweep,
  )
