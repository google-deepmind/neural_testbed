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

"""Exposing the public methods of generative data models."""

# Ensemble classification
from neural_testbed.generative.ensemble_gp_classification import GPClassificationEnsemble

# Neural tangents kernels
from neural_testbed.generative.nt_kernels import make_benchmark_kernel
from neural_testbed.generative.nt_kernels import make_linear_kernel

# Plotting
from neural_testbed.generative.plotting import sanity_1d
