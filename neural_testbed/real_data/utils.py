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

"""Utils for real data testbed."""

from typing import Dict, Tuple, Union, Optional

import chex
from neural_testbed import base as testbed_base
from neural_testbed.real_data import datasets
from neural_testbed.real_data import sweep
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

_DictFeat = Dict[str, Union[int, float]]
Features = Union[_DictFeat, chex.Array]


def _standardize_data(x: chex.Array,
                      axes: int = 0,
                      epsilon: float = 1e-8) -> chex.Array:
  """Returns standardized input."""
  mean, variance = tf.nn.moments(x, axes=axes)
  x_standardized = (x - mean) / tf.sqrt(variance + epsilon)
  return x_standardized


def _preprocess_structured_data(
    features: Features, label: int) -> testbed_base.Data:
  """Preprocess structured data into testbed standardized dictionary format."""
  # features can be a dict of numeric features or a single numeric feature
  if isinstance(features, Dict):
    features = tf.concat(
        [tf.cast(tf.expand_dims(x, -1), tf.float64) for x in features.values()],
        axis=0)
  else:
    features = tf.cast(features, tf.float64)

  features = _standardize_data(features)
  chex.assert_shape(features, (features.shape[0],))
  chex.assert_shape(label, ())
  label = tf.expand_dims(label, -1)

  return testbed_base.Data(x=features, y=label)


def _preprocess_image_data(dataset: Dict[str, chex.Array]) -> testbed_base.Data:
  """Preprocess image data into testbed standardized dictionary format."""
  assert 'image' in dataset
  assert 'label' in dataset
  image, label = dataset['image'], dataset['label']
  image = tf.image.convert_image_dtype(image, tf.float64)
  image = tf.reshape(image, [-1])
  image = _standardize_data(image)
  chex.assert_shape(image, (image.shape[0],))
  chex.assert_shape(label, ())
  label = tf.expand_dims(label, -1)

  return testbed_base.Data(x=image, y=label)


def _load_structured_dataset(
    dataset_info: datasets.DatasetInfo, split: str) -> testbed_base.Data:
  """Loads a structured dataset."""
  if split == 'train':
    data_split = f'train[:{dataset_info.num_train}]'
    batch_size = dataset_info.num_train
  else:
    data_split = f'train[-{dataset_info.num_test}:]'
    batch_size = dataset_info.num_test
  ds = tfds.load(
      name=dataset_info.dataset_name, split=data_split, as_supervised=True)
  ds = ds.map(_preprocess_structured_data)
  ds = ds.batch(batch_size)
  data = next(iter(tfds.as_numpy(ds)))

  return data


def _load_image_dataset(
    dataset_info: datasets.DatasetInfo, split: str) -> testbed_base.Data:
  """Loads an image dataset."""
  if split == 'train':
    data_split = f'train[:{dataset_info.num_train}]'
    batch_size = dataset_info.num_train
  else:
    data_split = split
    batch_size = dataset_info.num_test
  ds = tfds.load(
      name=dataset_info.dataset_name, split=data_split, with_info=False)
  ds = ds.map(_preprocess_image_data)
  ds = ds.batch(batch_size)
  data = next(iter(tfds.as_numpy(ds)))

  return data


def load_classification_dataset(
    dataset_info: datasets.DatasetInfo, split: str) -> testbed_base.Data:
  """Returns dataset data based on problem_config and split."""
  dataset_name = dataset_info.dataset_name
  if dataset_name not in datasets.CLASSIFICATION_DATASETS:
    raise ValueError(f'dataset {dataset_name} is not supported yet.')
  if dataset_name in datasets.STRUCTURED_DATASETS:
    return _load_structured_dataset(dataset_info=dataset_info, split=split)
  else:
    return _load_image_dataset(dataset_info=dataset_info, split=split)


def get_uci_data(name) -> Tuple[chex.Array, chex.Array]:
  """Returns an array of features and an array of labels for dataset `name`."""
  spec = datasets.DATA_SPECS.get(name)
  if spec is None:
    raise ValueError('Unknown dataset: {}. Available datasets:\n{}'.format(
        name, '\n'.join(datasets.DATA_SPECS.keys())))
  with tf.io.gfile.GFile(spec.path) as f:
    df = pd.read_csv(f)
  labels = df.pop(spec.label).to_numpy().astype(np.float32)
  for ex in spec.excluded:
    _ = df.pop(ex)
  features = df.to_numpy().astype(np.float32)
  return features, labels


def load_regression_dataset(
    dataset_name: str) -> Tuple[testbed_base.Data, testbed_base.Data]:
  """Returns dataset data from dataset_name."""
  if dataset_name not in datasets.REGRESSION_DATASETS:
    raise ValueError(f'dataset {dataset_name} is not supported yet.')
  x, y = get_uci_data(dataset_name)
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


def config_from_dataset_name(
    dataset_name: str,
    tau: int = 1,
    seed: int = 0,
    temperature: float = 0.01,
    noise_std: float = 1.,
    num_train: Optional[int] = None,
) -> sweep.ProblemConfig:
  """Returns a testbed problem based on a dataset name and tau.

  Args:
    dataset_name: name of the dataset.
    tau: value of tau for joint prediction.
    seed: random seed.
    temperature: temperature to be used in prior_knowledge. It does not affect
      the data.
    noise_std: noise_std to be used in prior_knowledge. It does not affect
      the data.
    num_train: optional, it can be used to limit the number of training data.

  Returns:
    A problem config for real data.
  """
  dataset_info = datasets.DATASETS_SETTINGS[dataset_name]
  if num_train is None:
    num_train = dataset_info.num_train
  else:
    num_train = min(num_train, dataset_info.num_train)

  prior_knowledge = testbed_base.PriorKnowledge(
      input_dim=dataset_info.input_dim,
      num_train=num_train,
      num_classes=dataset_info.num_classes,
      tau=tau,
      temperature=temperature,
      noise_std=noise_std,
  )

  problem_config = sweep.ProblemConfig(
      prior_knowledge=prior_knowledge,
      seed=seed,
      dataset_name=dataset_name,
  )

  return problem_config
