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

from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import baselines
from neural_testbed.agents.factories import bbb
from neural_testbed.agents.factories import deep_kernel
from neural_testbed.agents.factories import dropout
from neural_testbed.agents.factories import ensemble
from neural_testbed.agents.factories import ensemble_plus
from neural_testbed.agents.factories import epinet
from neural_testbed.agents.factories import hypermodel
from neural_testbed.agents.factories import knn
from neural_testbed.agents.factories import random_forest
from neural_testbed.agents.factories import sgmcmc
