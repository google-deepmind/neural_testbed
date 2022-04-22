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
"""Sweeps for agents, specialized to real datasets."""

import dataclasses
from typing import Sequence

import ml_collections
from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import baselines
from neural_testbed.agents.factories import bbb
from neural_testbed.agents.factories import dropout
from neural_testbed.agents.factories import ensemble
from neural_testbed.agents.factories import ensemble_plus
from neural_testbed.agents.factories import hypermodel
from neural_testbed.agents.factories import sgmcmc


def mlp_sweep() -> Sequence[baselines.MLPConfig]:
  """Sweep over hyperparams."""
  sweep = []
  for num_batches in [1000, 5000, 10_000]:
    for l2_weight_decay in [1e-2, 1e-1, 1, 10, 100]:
      sweep.append(baselines.MLPConfig(
          l2_weight_decay=l2_weight_decay,
          num_batches=num_batches,
          batch_strategy=False,
          adaptive_weight_scale=False,
      ))
  return tuple(sweep)


def mlp_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=baselines.MLPConfig(),
      ctor=baselines.make_mlp_agent,
      sweep=mlp_sweep,
  )


def dropout_sweep() -> Sequence[dropout.McDropoutConfig]:
  """Sweep over hyperparams."""
  sweep = []
  for num_batches in [1000, 5000, 10_000]:
    for length_scale in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
      for dropout_rate in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        sweep.append(dropout.McDropoutConfig(
            length_scale=length_scale,
            num_batches=num_batches,
            dropout_rate=dropout_rate,
            batch_strategy=False,
            adaptive_weight_scale=False,
        ))
  return tuple(sweep)


def dropout_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=dropout.McDropoutConfig(),
      ctor=dropout.make_mc_dropout_agent,
      sweep=dropout_sweep,
  )


def ensemble_sweep() -> Sequence[ensemble.VanillaEnsembleConfig]:
  """Sweep over hyperparams."""
  sweep = []
  for num_batches in [1000, 5000, 10_000]:
    for l2_weight_decay in [1e-2, 1e-1, 1, 10, 100, 1000]:
      sweep.append(ensemble.VanillaEnsembleConfig(
          l2_weight_decay=l2_weight_decay,
          num_batches=num_batches,
          num_ensemble=100,
          batch_strategy=False,
          adaptive_weight_scale=False,
      ))
  return tuple(sweep)


def ensemble_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=ensemble.VanillaEnsembleConfig(),
      ctor=ensemble.make_agent,
      sweep=ensemble_sweep,
  )


def ensemble_plus_sweep() -> Sequence[ensemble_plus.EnsembleConfig]:
  """Sweep over hyperparams."""
  sweep = []
  for num_batches in [1000, 5000, 10_000]:
    for l2_weight_decay in [1e-2, 1e-1, 1, 10, 100]:
      for prior_scale in [1, 3, 10, 30, 100]:
        sweep.append(ensemble_plus.EnsembleConfig(
            l2_weight_decay=l2_weight_decay,
            num_batches=num_batches,
            num_ensemble=100,
            prior_scale=prior_scale,
            batch_strategy=False,
            adaptive_weight_scale=False,
            temp_scale_prior='none',
        ))
  return tuple(sweep)


def ensemble_plus_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=ensemble_plus.EnsembleConfig(),
      ctor=ensemble_plus.make_agent,
      sweep=ensemble_plus_sweep,
  )


def hypermodel_sweep() -> Sequence[hypermodel.HypermodelConfig]:
  """Sweep over hyperparams."""
  sweep = []
  for num_batches in [1000, 5000, 10_000]:
    for l2_weight_decay in [1e-2, 1e-1, 1, 10, 100]:
      for prior_scale in [1, 3, 10, 30, 100]:
        sweep.append(hypermodel.HypermodelConfig(
            l2_weight_decay=l2_weight_decay,
            num_batches=num_batches,
            prior_hidden_sizes=(10, 10),
            prior_scale=prior_scale,
            index_dim=5,
            batch_strategy=False,
            adaptive_weight_scale=False,
            temp_scale_prior='none',
        ))
  return tuple(sweep)


def hypermodel_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=hypermodel.HypermodelConfig(),
      ctor=hypermodel.make_hypermodel_agent,
      sweep=hypermodel_sweep,
  )


def sgmcmc_sweep() -> Sequence[sgmcmc.SGMCMCConfig]:
  """Sweep over hyperparams."""
  sweep = []
  for learning_rate in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
    for prior_variance in [1, 3, 10, 30, 100, 300,]:
      for momentum_decay in [0, 0.9]:
        sweep.append(
            sgmcmc.SGMCMCConfig(
                learning_rate=learning_rate,
                prior_variance=prior_variance,
                momentum_decay=momentum_decay,
                adaptive_prior_variance=False))
  return tuple(sweep)


def sgmcmc_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=sgmcmc.SGMCMCConfig(),
      ctor=sgmcmc.make_agent,
      sweep=sgmcmc_sweep,
  )


def bbb_sweep() -> Sequence[bbb.BBBConfig]:
  """Sweep over hyperparams."""
  sweep = []
  for sigma_1 in [1, 2, 4]:
    for sigma_2 in [0.1, 0.3, 0.5, 0.7, 0.9]:
      for mixture_scale in [0, 0.5, 1]:
        for num_batches in [1000, 5000, 10_000]:
          sweep.append(
              bbb.BBBConfig(
                  sigma_1=sigma_1,
                  sigma_2=sigma_2,
                  mixture_scale=mixture_scale,
                  num_batches=num_batches,
                  learning_rate=1e-3,
                  adaptive_scale=False
              ))
  return tuple(sweep)


def bbb_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=bbb.BBBConfig(),
      ctor=bbb.make_agent,
      sweep=bbb_sweep,
  )

# Populate a global dictionary that makes these agents easy to access.
_AGENTS = {
    'baseline:uniform_class_probs': baselines.uniform_class_probs_paper_agent(),
    'mlp': mlp_agent(),
    'bbb': bbb_agent(),
    'ensemble': ensemble_agent(),
    'ensemble+': ensemble_plus_agent(),
    'hypermodel': hypermodel_agent(),
    'dropout': dropout_agent(),
    'sgmcmc': sgmcmc_agent(),
}


def get_implemented_agents() -> Sequence[str]:
  return list(_AGENTS.keys())


def get_paper_agent(agent: str) -> factories_base.PaperAgent:
  assert agent in get_implemented_agents()
  return _AGENTS[agent]


def dummy_config() -> ml_collections.ConfigDict:
  """Creates a dummy config with all possible components."""
  global_dict = {}
  for agent in get_implemented_agents():
    paper_agent = get_paper_agent(agent)
    global_dict.update(dataclasses.asdict(paper_agent.default))
  return ml_collections.ConfigDict(global_dict, type_safe=False)
