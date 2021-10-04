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
"""Exposing public methods for approximate posterior training."""

import dataclasses
from typing import Sequence

import ml_collections
from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import baselines
from neural_testbed.agents.factories import bbb
from neural_testbed.agents.factories import deep_kernel
from neural_testbed.agents.factories import dropout
from neural_testbed.agents.factories import ensemble
from neural_testbed.agents.factories import ensemble_plus
from neural_testbed.agents.factories import hypermodel
from neural_testbed.agents.factories import knn
from neural_testbed.agents.factories import random_forest
from neural_testbed.agents.factories import sgmcmc


# Populate a global dictionary that makes these agents easy to access.
_AGENTS = {
    'baseline:uniform_class_probs': baselines.uniform_class_probs_paper_agent(),
    'baseline:average_class_probs': baselines.average_class_probs_paper_agent(),
    'baseline:prior': baselines.prior_paper_agent(),
    'logistic_regression': baselines.logistic_regression_paper_agent(),
    'mlp': baselines.mlp_paper_agent(),
    'bbb': bbb.paper_agent(),
    'vanilla_ensemble': ensemble.paper_agent(),
    'ensemble_plus': ensemble_plus.paper_agent(),
    'hypermodel': hypermodel.paper_agent(),
    'mc_dropout': dropout.paper_agent(),
    'sgmcmc': sgmcmc.paper_agent(),
    'deep_kernel': deep_kernel.paper_agent(),
    'knn': knn.paper_agent(),
    'random_forest': random_forest.paper_agent(),
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
