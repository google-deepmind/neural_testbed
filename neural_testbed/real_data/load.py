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

"""Loading a leaderboard instance for the testbed."""

from neural_testbed import base as testbed_base
from neural_testbed.real_data import datasets
from neural_testbed.real_data import load_classification
from neural_testbed.real_data import load_regression


def problem_from_id(ds_name: str, tau: int = 1) -> testbed_base.TestbedProblem:
  """Load a classification problem from a real dataset specified by config."""
  if ds_name not in datasets.DATASETS_SETTINGS:
    raise ValueError(f'dataset={ds_name} is not supported')
  else:
    dataset_info = datasets.DATASETS_SETTINGS[ds_name]
  if ds_name in datasets.REGRESSION_DATASETS:
    assert tau == 1, 'High tau not implemented for regression yet.'
    return load_regression.load(dataset_info)
  else:
    return load_classification.load(dataset_info, tau)

