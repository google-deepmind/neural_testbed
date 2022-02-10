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

"""Defines the leaderboard sweep for RealData testbed."""

import dataclasses
from typing import Dict

from neural_testbed import base
from neural_testbed.real_data import data_sampler
from neural_testbed.real_data import datasets


# problem_ids are strings of the form {sweep_name}{SEPARATOR}{index}.
SEPARATOR = '/'


@dataclasses.dataclass(frozen=True)
class ProblemConfig:
  """Problem configuration including prior knowledge and some hyperparams."""
  # Agent's a priori knowledge about the problem.
  prior_knowledge: base.PriorKnowledge
  # Random seed controlling all the randomness in the problem.
  seed: int
  # Name of the dataset
  dataset_name: str
  # Test sampler constructor
  test_sampler_ctor: data_sampler.SamplerCtor = data_sampler.make_local_sampler
  # Number of inputs (X's) used for evaluation.
  num_test_seeds: int = 1000
  # Number of samples generated from ENN during evaluation.
  num_enn_samples: int = 1000

  @property
  def meta_data(self):
    meta = dataclasses.asdict(self)
    meta.pop('prior_knowledge')
    meta.update(dataclasses.asdict(self.prior_knowledge))
    return meta


def classification_sweep(num_seed: int = 5,
                         initial_seed: int = 0,
                         temperature: float = 0.01) -> Dict[str, ProblemConfig]:
  """Generate hyperparameter sweep for classification problems.

  Args:
    num_seed: number of seeds per configuration of other hyperparameters.
    initial_seed: initial value of the seed.
    temperature: temperature to be used in prior_knowledge. It does not affect
      the data.

  Returns:
    Mapping problem_id: gp_settings (for use in gp_load).
  """
  configs = []
  for tau in [1, 10]:
    for dataset_name in datasets.CLASSIFICATION_DATASETS:
      seed = initial_seed
      for unused_seed_inc in range(num_seed):
        seed += 1
        dataset_info = datasets.DATASETS_SETTINGS[dataset_name]
        prior_knowledge = base.PriorKnowledge(
            input_dim=dataset_info.input_dim,
            num_train=dataset_info.num_train,
            num_classes=dataset_info.num_classes,
            tau=tau,
            temperature=temperature,
        )

        configs.append(
            ProblemConfig(
                prior_knowledge=prior_knowledge,
                seed=seed,
                dataset_name=dataset_name,
            ))
  return {f'classification{SEPARATOR}{i}': v for i, v in enumerate(configs)}


def classification_num_data_sweep(
    num_seed: int = 5,
    initial_seed: int = 0,
    temperature: float = 0.01,
) -> Dict[str, ProblemConfig]:
  """Generate hyperparameter sweep for classification problems with different number of training data.

  Args:
    num_seed: number of seeds per configuration of other hyperparameters.
    initial_seed: initial value of the seed.
    temperature: temperature to be used in prior_knowledge. It does not affect
      the data.

  Returns:
    Mapping problem_id: gp_settings (for use in gp_load).
  """
  configs = []
  for tau in [1, 10]:
    for dataset_name in datasets.CLASSIFICATION_DATASETS:
      seed = initial_seed
      for num_train in [1, 10, 100, 1000, 10_000, 100_000]:
        for unused_seed_inc in range(num_seed):
          seed += 1
          dataset_info = datasets.DATASETS_SETTINGS[dataset_name]
          # Update num_train of the dataset
          num_train = min(num_train, dataset_info.num_train)

          prior_knowledge = base.PriorKnowledge(
              input_dim=dataset_info.input_dim,
              num_train=num_train,
              num_classes=dataset_info.num_classes,
              tau=tau,
              temperature=temperature,
          )

          configs.append(
              ProblemConfig(
                  prior_knowledge=prior_knowledge,
                  seed=seed,
                  dataset_name=dataset_name,
              ))
  return {
      f'classification_variant_data{SEPARATOR}{i}': v
      for i, v in enumerate(configs)
  }


def regression_sweep(num_seed: int = 5,
                     initial_seed: int = 0,
                     noise_std: float = 1.) -> Dict[str, ProblemConfig]:
  """Generate hyperparameter sweep for regression problems.

  Args:
    num_seed: number of seeds per configuration of other hyperparameters.
    initial_seed: initial value of the seed.
    noise_std: noise_std to be used in prior_knowledge. It does not affect
      the data.

  Returns:
    Mapping problem_id: gp_settings (for use in gp_load).
  """
  configs = []
  for dataset_name in datasets.REGRESSION_DATASETS:
    seed = initial_seed
    for unused_seed_inc in range(num_seed):
      seed += 1
      dataset_info = datasets.DATASETS_SETTINGS[dataset_name]
      prior_knowledge = base.PriorKnowledge(
          input_dim=dataset_info.input_dim,
          num_train=dataset_info.num_train,
          num_classes=dataset_info.num_classes,
          tau=1,
          noise_std=noise_std,
      )

      configs.append(
          ProblemConfig(
              prior_knowledge=prior_knowledge,
              seed=seed,
              dataset_name=dataset_name,
          ))
  return {f'regression{SEPARATOR}{i}': v for i, v in enumerate(configs)}


SETTINGS = {**regression_sweep(),
            **classification_sweep(),
            **classification_num_data_sweep(),}

REGRESSION = tuple(regression_sweep().keys())
CLASSIFICATION = tuple(classification_sweep().keys())
CLASSIFICATION_NUM_DATA = tuple(classification_num_data_sweep().keys())
