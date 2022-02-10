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

"""Storing the name of supported datasets and their information."""
import dataclasses
import os
from typing import List


@dataclasses.dataclass
class DatasetInfo:
  dataset_name: str
  num_classes: int
  input_dim: int
  num_train: int
  num_test: int


def regression_datasets():
  """Returns a dictionary of regression datasets currently supported."""
  datasets = {
      'boston_housing':
          DatasetInfo(
              dataset_name='boston_housing',
              num_classes=1,
              input_dim=13,
              num_train=202,
              num_test=51),
      'concrete_strength':
          DatasetInfo(
              dataset_name='concrete_strength',
              num_classes=1,
              input_dim=8,
              num_train=824,
              num_test=206),
      'energy_efficiency':
          DatasetInfo(
              dataset_name='energy_efficiency',
              num_classes=1,
              input_dim=8,
              num_train=614,
              num_test=154),
      'naval_propulsion':
          DatasetInfo(
              dataset_name='naval_propulsion',
              num_classes=1,
              input_dim=16,
              num_train=9547,
              num_test=2387),
      'kin8nm':
          DatasetInfo(
              dataset_name='kin8nm',
              num_classes=1,
              input_dim=8,
              num_train=6553,
              num_test=1639),
      'power_plant':
          DatasetInfo(
              dataset_name='power_plant',
              num_classes=1,
              input_dim=4,
              num_train=7654,
              num_test=1914),
      'protein_structure':
          DatasetInfo(
              dataset_name='protein_structure',
              num_classes=1,
              input_dim=9,
              num_train=36584,
              num_test=9146),
      'wine':
          DatasetInfo(
              dataset_name='wine',
              num_classes=1,
              input_dim=11,
              num_train=1279,
              num_test=320),
      'yacht_hydrodynamics':
          DatasetInfo(
              dataset_name='yacht_hydrodynamics',
              num_classes=1,
              input_dim=7,
              num_train=246,
              num_test=62)
  }
  return datasets


def structured_datasets():
  """Returns a dictionary of structured datasets currently supported."""
  datasets = {
      'iris':
          DatasetInfo(
              dataset_name='iris',
              num_classes=3,
              input_dim=4,
              num_train=120,
              num_test=30),
      'wine_quality':
          DatasetInfo(
              dataset_name='wine_quality',
              num_classes=11,
              input_dim=11,
              num_train=3918,
              num_test=980),
      'german_credit_numeric':
          DatasetInfo(
              dataset_name='german_credit_numeric',
              num_classes=2,
              input_dim=24,
              num_train=800,
              num_test=200),
  }
  return datasets


def image_datasets():
  """Returns a dictionary of image datasets currently supported."""
  dataset = {
      'cmaterdb':
          DatasetInfo(
              dataset_name='cmaterdb',
              num_classes=10,
              input_dim=3_072,
              num_train=5_000,
              num_test=1_000),
      'mnist':
          DatasetInfo(
              dataset_name='mnist',
              num_classes=10,
              input_dim=784,
              num_train=60_000,
              num_test=10_000),
      'emnist/digits':
          DatasetInfo(
              dataset_name='emnist/digits',
              num_classes=10,
              input_dim=784,
              num_train=240_000,
              num_test=40_000),
      'emnist/letters':
          DatasetInfo(
              dataset_name='emnist/letters',
              num_classes=37,
              input_dim=784,
              num_train=88_800,
              num_test=14_800),
      'fashion_mnist':
          DatasetInfo(
              dataset_name='fashion_mnist',
              num_classes=10,
              input_dim=784,
              num_train=60_000,
              num_test=10_000),
      'mnist_corrupted/shot_noise':
          DatasetInfo(
              dataset_name='mnist_corrupted/shot_noise',
              num_classes=10,
              input_dim=784,
              num_train=60_000,
              num_test=10_000),
      'cifar10':
          DatasetInfo(
              dataset_name='cifar10',
              num_classes=10,
              input_dim=3_072,
              num_train=50_000,
              num_test=10_000),
  }
  return dataset


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


STRUCTURED_DATASETS = tuple(structured_datasets().keys())
IMAGE_DATASETS = tuple(image_datasets().keys())
CLASSIFICATION_DATASETS = STRUCTURED_DATASETS + IMAGE_DATASETS

REGRESSION_DATASETS = tuple(regression_datasets().keys())

DATASETS_SETTINGS = {
    **image_datasets(),
    **structured_datasets(),
    **regression_datasets(),
}

DATASETS = CLASSIFICATION_DATASETS + REGRESSION_DATASETS
