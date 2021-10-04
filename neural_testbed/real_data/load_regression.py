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

"""Loading a regression real data problem for the testbed."""

import dataclasses
import os
from typing import List, Tuple

import chex
import jax
from neural_testbed import base as testbed_base
from neural_testbed import likelihood
from neural_testbed.real_data import data_sampler as real_data
from neural_testbed.real_data import datasets
import numpy as np
import pandas as pd
import tensorflow as tf


@dataclasses.dataclass
class UCIDataSpec:
  path: str
  desc: str
  label: str
  excluded: List[str]


# TODO(author3): Avoid hard-coding directory string so it's user-specified.
UCI_BASE_DIR = '/tmp/uci_datasets'
# BEGIN GOOGLE-INTERNAL
UCI_BASE_DIR = '/path_to_regression_dataset_repo/distbelief/uci_datasets/'
# END GOOGLE-INTERNAL
DATA_SPECS = {
    'boston_housing': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'boston_housing.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='MEDV',
        excluded=[]),
    'concrete_strength': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'concrete_strength.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='concrete_compressive_strength',
        excluded=[]),
    'energy_efficiency': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'energy_efficiency.csv'),
        desc=('This study looked into assessing the heating load and cooling '
              'load requirements of buildings (that is, energy efficiency) as '
              'a function of building parameters. **Heating load only**.'),
        label='Y1',
        excluded=['Y2']),
    'naval_propulsion': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'naval_propulsion.csv'),
        desc=('Data have been generated from a sophisticated simulator of a '
              'Gas Turbines (GT), mounted on a Frigate characterized by a '
              'Combined Diesel eLectric And Gas (CODLAG) propulsion plant '
              'type. **GT Turbine decay state coefficient only**'),
        label='GT Turbine decay state coefficient',
        excluded=['GT Compressor decay state coefficient']),
    'kin8nm': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'kin8nm.csv'),
        desc=('This is data set is concerned with the forward kinematics of '
              'an 8 link robot arm. Among the existing variants of this data '
              'set we have used the variant 8nm, which is known to be highly '
              'non-linear and medium noisy.'),
        label='y',
        excluded=[]),
    'power_plant': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'power_plant.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='PE',
        excluded=[]),
    'protein_structure': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'protein_structure.csv'),
        desc=('This is a data set of Physicochemical Properties of Protein '
              'Tertiary Structure. The data set is taken from CASP 5-9. There '
              'are 45730 decoys and size varying from 0 to 21 armstrong.'),
        label='RMSD',
        excluded=[]),
    'wine': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'wine.csv'),
        desc=('The dataset is related to red variant of the Portuguese '
              '"Vinho Verde" wine. **NB contains red wine examples only**'),
        label='quality',
        excluded=[]),
    'yacht_hydrodynamics': UCIDataSpec(
        path=os.path.join(UCI_BASE_DIR, 'yacht_hydrodynamics.csv'),
        desc=('Delft data set, used to predict the hydodynamic performance of '
              'sailing yachts from dimensions and velocity.'),
        label='Residuary resistance per unit weight of displacement',
        excluded=[])
}


def get_uci_data(name) -> Tuple[chex.Array, chex.Array]:
  """Returns an array of features and an array of labels for dataset `name`."""
  spec = DATA_SPECS.get(name)
  if spec is None:
    raise ValueError('Unknown dataset: {}. Available datasets:\n{}'.format(
        name, '\n'.join(DATA_SPECS.keys())))
  with tf.io.gfile.GFile(spec.path) as f:
    df = pd.read_csv(f)
  labels = df.pop(spec.label).to_numpy().astype(np.float32)
  for ex in spec.excluded:
    _ = df.pop(ex)
  features = df.to_numpy().astype(np.float32)
  return features, labels


def load_dataset(name) -> Tuple[testbed_base.Data, testbed_base.Data]:
  """Loads train and test data for dataset `name`."""
  x, y = get_uci_data(name)
  if len(y.shape) == 1:
    y = y[:, None]
  train_test_split = 0.8
  random_permutation = np.random.permutation(x.shape[0])
  n_train = int(x.shape[0] * train_test_split)
  train_ind = random_permutation[:n_train]
  test_ind = random_permutation[n_train:]
  x_train, y_train = x[train_ind, :], y[train_ind, :]
  x_test, y_test = x[test_ind, :], y[test_ind, :]

  x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
  y_mean = np.mean(y_train, axis=0)
  epsilon = tf.keras.backend.epsilon()
  x_train = (x_train - x_mean) / (x_std + epsilon)
  x_test = (x_test - x_mean) / (x_std + epsilon)
  y_train, y_test = y_train - y_mean, y_test - y_mean
  train_data = testbed_base.Data(x=x_train, y=y_train)
  test_data = testbed_base.Data(x=x_test, y=y_test)
  return train_data, test_data


def load(dataset_info: datasets.DatasetInfo) -> testbed_base.TestbedProblem:
  """Load a regression problem from a real dataset specified by config."""
  num_enn_samples = 1000  # We set it to the number we use for our testbed
  train_data, test_data = load_dataset(name=dataset_info.dataset_name)

  data_sampler = real_data.RealDataSampler(train_data, test_data)

  prior_knowledge = testbed_base.PriorKnowledge(
      input_dim=dataset_info.input_dim,
      num_train=dataset_info.num_train,
      num_classes=dataset_info.num_classes,
      noise_std=1,
      tau=1)

  sample_based_kl = likelihood.GaussianSmoothedSampleKL(
      num_test_seeds=dataset_info.num_test,
      num_enn_samples=num_enn_samples,
      enn_sigma=1.,
      key=jax.random.PRNGKey(0),
  )

  return likelihood.SampleBasedTestbed(
      data_sampler, sample_based_kl, prior_knowledge)
