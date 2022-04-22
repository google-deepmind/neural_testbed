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

"""Exposing the public methods of real data."""

# Realdata data sampler
from neural_testbed.real_data.data_sampler import make_global_sampler
from neural_testbed.real_data.data_sampler import make_local_sampler
from neural_testbed.real_data.data_sampler import RealDataSampler

# Realdata datasets
from neural_testbed.real_data.datasets import CLASSIFICATION_DATASETS
from neural_testbed.real_data.datasets import DatasetInfo
from neural_testbed.real_data.datasets import DATASETS
from neural_testbed.real_data.datasets import DATASETS_SETTINGS
from neural_testbed.real_data.datasets import REGRESSION_DATASETS

# Realdata loading of testbed problem
from neural_testbed.real_data.load import problem_from_config
from neural_testbed.real_data.load import problem_from_id

# Realdata sweep of testbed problems
from neural_testbed.real_data.sweep import CLASSIFICATION
from neural_testbed.real_data.sweep import CLASSIFICATION_NUM_DATA
from neural_testbed.real_data.sweep import ProblemConfig
from neural_testbed.real_data.sweep import REGRESSION
from neural_testbed.real_data.sweep import SETTINGS

# Realdata utils
from neural_testbed.real_data.utils import config_from_dataset_name
from neural_testbed.real_data.utils import load_classification_dataset
from neural_testbed.real_data.utils import load_regression_dataset
