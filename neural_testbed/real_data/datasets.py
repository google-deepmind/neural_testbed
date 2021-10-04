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
from typing import Dict


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


def original_structured_datasets():
  """Returns a dictionary of original structured datasets currently supported."""
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


def original_image_datasets():
  """Returns a dictionary of original image datasets currently supported."""
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


def data_variant_datasets(
    datasets: Dict[str, DatasetInfo]) -> Dict[str, DatasetInfo]:
  """Returns real datasets with original and limited numbers of train data."""
  sweep = {**datasets}
  for ds_name, info in datasets.items():
    for num_train in [1, 10, 100, 1000, 10_000]:
      num_train = min(num_train, info.num_train)
      data_limited_ds_name = ds_name + f'-num_train={num_train}'
      data_limited_info = DatasetInfo(
          dataset_name=info.dataset_name,
          num_classes=info.num_classes,
          input_dim=info.input_dim,
          num_train=num_train,
          num_test=info.num_test)
      sweep[data_limited_ds_name] = data_limited_info

  return sweep


ORIGINAL_STRUCTURED_DATASETS = tuple(original_structured_datasets().keys())
ORIGINAL_IMAGE_DATASETS = tuple(original_image_datasets().keys())
ORIGINAL_CLASSIFICATION_DATASETS = ORIGINAL_STRUCTURED_DATASETS + ORIGINAL_IMAGE_DATASETS

STRUCTURED_DATASETS_SETTINGS = data_variant_datasets(
    original_structured_datasets())
STRUCTURED_DATASETS = tuple(STRUCTURED_DATASETS_SETTINGS.keys())
IMAGE_DATASETS_SETTINGS = data_variant_datasets(original_image_datasets())
IMAGE_DATASETS = tuple(IMAGE_DATASETS_SETTINGS.keys())
CLASSIFICATION_DATASETS = STRUCTURED_DATASETS + IMAGE_DATASETS

REGRESSION_DATASETS = tuple(regression_datasets().keys())
REGRESSION_DATASETS_SETTINGS = regression_datasets()


DATASETS_SETTINGS = {
    **STRUCTURED_DATASETS_SETTINGS,
    **IMAGE_DATASETS_SETTINGS,
    **REGRESSION_DATASETS_SETTINGS,
}
DATASETS = tuple(DATASETS_SETTINGS.keys())
