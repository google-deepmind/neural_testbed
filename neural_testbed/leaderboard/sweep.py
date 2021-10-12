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

"""Defines the leaderboard sweep for GP testbed."""

import dataclasses
from typing import Callable, Dict, List, Sequence

from neural_testbed import base


# problem_ids are strings of the form {sweep_name}{SEPARATOR}{index}.
SEPARATOR = '/'
# DataFrame results are saved to this name in the log.
DATAFRAME = 'neural_testbed_5'


@dataclasses.dataclass(frozen=True)
class ProblemConfig:
  """Problem configuration including prior knowledge and some hyperparams."""
  # Agent's a priori knowledge about the problem.
  prior_knowledge: base.PriorKnowledge
  # Random seed controlling all the randomness in the problem.
  seed: int
  # Number of inputs (X's) used for evaluation.
  num_test_seeds: int = 1000
  # Number of samples generated from ENN during evaluation.
  num_enn_samples: int = 1000
  # Number of inputs (X's) cached for evaluation.
  num_test_cache: int = 1000  # Used only by GPRegression data_sampler.
  epistemic_only: bool = False  # Used only by GPRegression.

  @property
  def meta_data(self):
    meta = dataclasses.asdict(self)
    meta.pop('prior_knowledge')
    meta.update(dataclasses.asdict(self.prior_knowledge))
    return meta


def regression_sweep(num_seed: int = 10,
                     initial_seed: int = 0) -> Dict[str, ProblemConfig]:
  """Generate hyperparameter sweep for regression.

  Args:
    num_seed: number of seeds per configuratioon of other hyperparameters.
    initial_seed: initial value of the seed.
  Returns:
    Mapping problem_id: gp_settings (for use in gp_load).
  """

  configs = []
  # TODO(author2): convert to itertools
  seed = initial_seed
  for input_dim in [1, 10, 100]:
    for data_ratio in [1, 10, 100]:
      for noise_std in [0.01, 0.1, 1]:
        for unused_seed_inc in range(num_seed):
          seed += 1
          num_train = int(data_ratio * input_dim)
          prior_knowledge = base.PriorKnowledge(
              input_dim=input_dim,
              num_train=num_train,
              noise_std=noise_std,
              num_classes=1,  # Currently fixed and not part of the configs.
              tau=1,  # Currently regression only supports tau=1
              layers=1,
              )
          configs.append(ProblemConfig(prior_knowledge, seed,
                                       num_enn_samples=100))
  return {f'regression{SEPARATOR}{i}': v for i, v in enumerate(configs)}


def regression_test_sweep() -> Dict[str, ProblemConfig]:
  """Reduced sweep for testing regression."""
  full_configs = list(regression_sweep(num_seed=1).values())
  configs = _filter_unique_configs(
      full_configs,
      lambda x: ((x.prior_knowledge.noise_std == 0.1)  # pylint: disable=g-long-lambda
                 and (x.prior_knowledge.input_dim == 10)
                 and (x.prior_knowledge.num_train == 10)
                 and (x.prior_knowledge.tau == 1)))
  return {f'regression_test{SEPARATOR}{i}': v for i, v in enumerate(configs)}


def classification_2d_sweep(num_seed: int = 10,
                            initial_seed: int = 0) -> Dict[str, ProblemConfig]:
  """Generate hyperparameter sweep for 2d classification problems.

  Args:
    num_seed: number of seeds per configuratioon of other hyperparameters.
    initial_seed: initial value of the seed.
  Returns:
    Mapping problem_id: gp_settings (for use in gp_load).
  """
  configs = []
  # TODO(author2): convert to itertools
  seed = initial_seed
  for num_train in [1, 3, 10, 30, 100, 300, 1000]:
    for temperature in [0.01, 0.1, 0.5]:
      for unused_seed_inc in range(num_seed):
        for tau in [1, 100]:
          seed += 1

          prior_knowledge = base.PriorKnowledge(
              input_dim=2,
              num_train=num_train,
              num_classes=2,  # Currently fixed and not part of the configs.
              tau=tau,
              layers=2,
              temperature=temperature,
              )

          configs.append(ProblemConfig(prior_knowledge, seed))
  return {f'classification_2d{SEPARATOR}{i}': v
          for i, v in enumerate(configs)}


def classification_2d_test_sweep() -> Dict[str, ProblemConfig]:
  """Reduced sweep for testing 2d classification problems."""
  full_configs = list(classification_2d_sweep(num_seed=1).values())
  configs = _filter_unique_configs(
      full_configs,
      lambda x: ((x.prior_knowledge.temperature == 0.01)  # pylint: disable=g-long-lambda
                 and (x.prior_knowledge.num_train == 10)
                 and (x.prior_knowledge.tau == 1))
      )
  return {f'classification_2d_test{SEPARATOR}{i}':
              v for i, v in enumerate(configs)}


def classification_2d_light_sweep() -> Dict[str, ProblemConfig]:
  """Reduced num seed for agent hyperparameter optimization classification."""
  configs = list(classification_2d_sweep(num_seed=3).values())
  return {f'classification_2d_light{SEPARATOR}{i}':
              v for i, v in enumerate(configs)}


def enn_paper_sweep() -> Dict[str, ProblemConfig]:
  """Generates sweep for GP regression in ENN paper."""
  configs = list(regression_sweep().values())
  return {f'enn_paper{SEPARATOR}{i}': dataclasses.replace(problem_config,
                                                          epistemic_only=True)
          for i, problem_config in enumerate(configs)}


def enn_paper_test_sweep() -> Dict[str, ProblemConfig]:
  """Reduced sweep for testing ENN paper."""
  full_configs = list(regression_sweep(num_seed=1).values())
  configs = _filter_unique_configs(full_configs,
                                   lambda x: x.prior_knowledge.noise_std == .1)
  return {f'enn_paper_test{SEPARATOR}{i}':
              dataclasses.replace(problem_config, epistemic_only=True)
          for i, problem_config in enumerate(configs)}


def _filter_unique_configs(
    configs: Sequence[ProblemConfig],
    filter_fn: Callable[[ProblemConfig], bool] = lambda _: True,
) -> List[ProblemConfig]:  # pytype: disable=annotation-type-mismatch
  """Filters a list of problem_config to their unique occurrences for testing.

  Args:
    configs: list of ProblemConfig.
    filter_fn: optional function to apply only to subset meeting this condition.

  Returns:
    List of unique occurrences for testing.
  """
  observed_configs = set()
  new_configs = []
  for problem_config in configs:
    if filter_fn(problem_config):
      if problem_config not in observed_configs:
        new_configs.append(problem_config)
        observed_configs.add(problem_config)
  return new_configs


SETTINGS = {
    **regression_sweep(),
    **regression_test_sweep(),
    **enn_paper_sweep(),
    **enn_paper_test_sweep(),
    **classification_2d_sweep(),
    **classification_2d_test_sweep(),
    **classification_2d_light_sweep(),}
REGRESSION = tuple(regression_sweep().keys())
REGRESSION_TEST = tuple(regression_test_sweep().keys())
ENN_PAPER = tuple(enn_paper_sweep().keys())
ENN_PAPER_TEST = tuple(enn_paper_test_sweep().keys())
CLASSIFICATION_2D = tuple(classification_2d_sweep().keys())
CLASSIFICATION_2D_LIGHT = tuple(classification_2d_light_sweep().keys())
CLASSIFICATION_2D_TEST = tuple(classification_2d_test_sweep().keys())
