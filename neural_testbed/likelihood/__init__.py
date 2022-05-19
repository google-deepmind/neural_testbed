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

"""Exposing the public methods of likelihood."""

# TODO(author2): Work on reducing this public scope

# Base classes
from neural_testbed.likelihood.base import GenerativeDataSampler
from neural_testbed.likelihood.base import SampleBasedKL
from neural_testbed.likelihood.base import SampleBasedTestbed


# Classification
from neural_testbed.likelihood.classification import add_classification_accuracy_ece
from neural_testbed.likelihood.classification import CategoricalKLSampledXSampledY
from neural_testbed.likelihood.classification import ClassificationSampleAccEce
from neural_testbed.likelihood.classification import compute_discrete_kl

# Classification Projection
from neural_testbed.likelihood.classification_projection import CategoricalClusterKL
from neural_testbed.likelihood.classification_projection import JointLLCalculatorProjection
from neural_testbed.likelihood.classification_projection import KmeansCluster
from neural_testbed.likelihood.classification_projection import RandomProjection


# Regression
from neural_testbed.likelihood.regression import gaussian_log_likelihood
from neural_testbed.likelihood.regression import GaussianSampleKL
from neural_testbed.likelihood.regression import GaussianSmoothedSampleKL
from neural_testbed.likelihood.regression import optimized_gaussian_ll
