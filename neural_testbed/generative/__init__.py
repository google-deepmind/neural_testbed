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

# Classification w.r.t environment likelihood
from neural_testbed.generative.classification_envlikelihood import ClassificationEnvLikelihood
from neural_testbed.generative.classification_envlikelihood import make_gaussian_sampler
from neural_testbed.generative.classification_envlikelihood import make_local_sampler
from neural_testbed.generative.classification_envlikelihood import make_weibull_sampler
from neural_testbed.generative.classification_envlikelihood import XGenerator

# Factories
from neural_testbed.generative.factories import make_2layer_mlp_logit_fn
from neural_testbed.generative.factories import make_filtered_gaussian_data

# Classification with GP likelihood
from neural_testbed.generative.gp_classification_envlikelihood import GPClassificationEnvLikelihood

# Regression
from neural_testbed.generative.gp_regression import GPRegression
from neural_testbed.generative.gp_regression import TestbedGPRegression

# Regression w.r.t environment likelihood
from neural_testbed.generative.gp_regression_envlikelihood import GPRegressionEnvLikelihood

# Neural tangents kernels
from neural_testbed.generative.nt_kernels import KernelCtor
from neural_testbed.generative.nt_kernels import make_benchmark_kernel
from neural_testbed.generative.nt_kernels import make_linear_kernel
from neural_testbed.generative.nt_kernels import MLPKernelCtor

# Plotting
from neural_testbed.generative.plotting import generate_2d_plots
from neural_testbed.generative.plotting import sanity_1d
from neural_testbed.generative.plotting import sanity_plots
