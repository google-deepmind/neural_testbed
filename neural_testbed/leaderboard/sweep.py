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

from typing import Any, Dict


# gp_ids are strings of the form {sweep_name}{SEPARATOR}{index}.
SEPARATOR = '/'
# DataFrame results are saved to this name in XData
DATAFRAME = 'neural_testbed_1'


def classification_sweep() -> Dict[str, Dict[str, Any]]:
  """Generates sweep for GP classification.

  Returns:
    Mapping gp_id: gp_settings (for use in gp_load).
  """
  hypers = []
  # TODO(author2): convert to itertools
  for num_train in [1, 10, 100, 1000]:
    for input_dim in [1, 10, 100, 1000]:
      for seed in range(10):
        for temperature in [0.1, 1]:
          hypers.append({
              'num_train': num_train,
              'input_dim': input_dim,
              'seed': seed,
              'temperature': temperature,
          })
  return {f'classification{SEPARATOR}{i}': v for i, v in enumerate(hypers)}


def classification_test_sweep() -> Dict[str, Dict[str, Any]]:
  """Reduced sweep for testing classification."""
  hypers = []
  # TODO(author2): convert to itertools
  for num_train in [1, 10]:
    for input_dim in [1, 10]:
      for seed in range(1):
        for temperature in [1]:
          hypers.append({
              'num_train': num_train,
              'input_dim': input_dim,
              'seed': seed,
              'temperature': temperature,
          })
  return {f'classification_test{SEPARATOR}{i}': v for i, v in enumerate(hypers)}


def regression_sweep() -> Dict[str, Dict[str, Any]]:
  """Generates sweep for 1D GP regression.

  Returns:
    Mapping gp_id: gp_settings (for use in gp_load).
  """
  hypers = []
  # TODO(author2): convert to itertools
  for num_train in [1, 10, 100, 1000]:
    for input_dim in [1, 10, 100, 1000]:
      for seed in range(10):
        for noise_std in [0.01, 0.1, 1]:
          hypers.append({
              'num_train': num_train,
              'input_dim': input_dim,
              'seed': seed,
              'noise_std': noise_std,
          })
  return {f'regression{SEPARATOR}{i}': v for i, v in enumerate(hypers)}


SETTINGS = {**classification_sweep(), **classification_test_sweep(),
            **regression_sweep()}
CLASSIFICATION = tuple(classification_sweep().keys())
CLASSIFICATION_TEST = tuple(classification_test_sweep().keys())
REGRESSION = tuple(regression_sweep().keys())
