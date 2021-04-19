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

from typing import Dict, Optional, Tuple, Union

import chex
import dataclasses
from enn import base as enn_base
from neural_testbed import base as testbed_base
from neural_testbed import likelihood
from neural_testbed.real_data import real_data_classification as real_data
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

RealDataProblem = Tuple[likelihood.SampleBasedTestbed, enn_base.BatchIterator,
                        Optional[Dict[str, enn_base.BatchIterator]]]
Features = Union[Dict[str, Union[int, float]], chex.Array]


@dataclasses.dataclass
class DatasetInfo:
  dataset_name: str
  num_classes: int
  input_dim: int
  num_train: int
  num_test: int


SUPPORTED_DATASETS = frozenset([
    'iris',
    'wine_quality',
    'german_credit_numeric',
    'cmaterdb',
    'mnist',
    'emnist/digits',
    'emnist/letters',
    'fashion_mnist',
    'mnist_corrupted/shot_noise',
    'cifar10'
])
STRUCTURED_DATASETS = {
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


@dataclasses.dataclass
class DatasetConfig:
  dataset_name: str
  train_batch_size: int = 1000
  eval_batch_size: int = 1000
  with_eval: bool = True


def _standardize_data(x: chex.Array,
                      axes: int = 0,
                      epsilon: float = 1e-8) -> chex.Array:
  """Returns standardized input."""
  mean, variance = tf.nn.moments(x, axes=axes)
  x_standardized = (x - mean) / tf.sqrt(variance + epsilon)
  return x_standardized


def _preprocess_structured_data(
    data_index: int, dataset: Tuple[Features,
                                    int]) -> Dict[str, chex.Array]:
  """Preprocess structured data into testbed standardized dictionary format.

  Args:
    data_index: An integer which is unique for each data sample.
    dataset: A tuple of features and label for a data sample. Features can be a
      dict of numeric (single int/float) features or a single array feature.

  Returns:
    A dict in the testbed standardized dictionary format.
  """
  features, label = dataset
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
  chex.assert_shape(data_index, ())
  label = tf.expand_dims(label, -1)
  data_index = tf.expand_dims(data_index, -1)
  chex.assert_shape(label, (1,))
  chex.assert_shape(data_index, (1,))

  return {'x': features, 'y': label, 'label': label, 'data_index': data_index}


def _preprocess_image_data(
    data_index: int, dataset: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
  """Preprocess image data into testbed standardized dictionary format."""
  assert 'image' in dataset
  assert 'label' in dataset
  image, label = dataset['image'], dataset['label']
  image = tf.image.convert_image_dtype(image, tf.float64)
  image = tf.reshape(image, [-1])
  image = _standardize_data(image)
  chex.assert_shape(image, (image.shape[0],))
  chex.assert_shape(label, ())
  chex.assert_shape(data_index, ())
  label = tf.expand_dims(label, -1)
  data_index = tf.expand_dims(data_index, -1)
  chex.assert_shape(label, (1,))
  chex.assert_shape(data_index, (1,))

  return {'x': image, 'y': label, 'label': label, 'data_index': data_index}


def _load_structured_dataset(
    name: str, split: str, batch_size: int,
    is_training: bool) -> Tuple[enn_base.BatchIterator, DatasetInfo]:
  """Loads a structured dataset."""
  if split == 'train':
    data_split = 'train[:80%]'
  else:
    data_split = 'train[80%:]'
  ds = tfds.load(name=name, split=data_split, as_supervised=True)
  ds = ds.enumerate()
  ds = ds.map(_preprocess_structured_data)
  if is_training:
    ds = ds.shuffle(50 * batch_size)
  ds = ds.cache().repeat()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(1)

  dataset_info = STRUCTURED_DATASETS[name]
  return iter(tfds.as_numpy(ds)), dataset_info


def _load_image_dataset(
    name: str, split: str, batch_size: int,
    is_training: bool) -> Tuple[enn_base.BatchIterator, DatasetInfo]:
  """Loads an image dataset."""
  ds, info = tfds.load(name=name, split=split, with_info=True)
  ds = ds.enumerate()
  ds = ds.map(_preprocess_image_data)
  if is_training:
    ds = ds.shuffle(50 * batch_size)
  ds = ds.cache().repeat()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(1)

  num_classes = info.features['label'].num_classes
  input_dim = np.prod(info.features['image'].shape)
  num_train = info.splits['train'].num_examples
  num_test = info.splits['test'].num_examples
  dataset_info = DatasetInfo(
      dataset_name=name,
      num_classes=num_classes,
      input_dim=input_dim,
      num_train=num_train,
      num_test=num_test)
  return iter(tfds.as_numpy(ds)), dataset_info


def load_dataset(
    name: str, split: str, batch_size: int,
    is_training: bool) -> Tuple[enn_base.BatchIterator, DatasetInfo]:
  """Loads the dataset as an iterator of batches."""
  if name not in SUPPORTED_DATASETS:
    raise ValueError(f'dataset {name} is not supported yet.')
  if name in STRUCTURED_DATASETS:
    return _load_structured_dataset(
        name=name, split=split, batch_size=batch_size, is_training=is_training)
  else:
    return _load_image_dataset(
        name=name, split=split, batch_size=batch_size, is_training=is_training)


def _testbed_from_config(
    config: DatasetConfig) -> likelihood.SampleBasedTestbed:
  """Return a testbed problem from a real dataset specified by config."""
  train_iter, info = load_dataset(
      name=config.dataset_name,
      split='train',
      batch_size=config.train_batch_size,
      is_training=False)
  test_iter, _ = load_dataset(
      name=config.dataset_name,
      split='test',
      batch_size=info.num_test,
      is_training=False)

  data_sampler = real_data.RealDataClassification(train_iter=train_iter,
                                                  test_iter=test_iter)

  sample_based_kl = likelihood.CategoricalSampleKL(
      num_test_seeds=info.num_test,
      num_enn_samples=100,
  )
  sample_based_kl = likelihood.add_classification_accuracy(sample_based_kl)
  prior_knowledge = testbed_base.PriorKnowledge(
      input_dim=info.input_dim,
      num_train=info.num_train,
      num_classes=info.num_classes
  )
  return likelihood.SampleBasedTestbed(
      data_sampler, sample_based_kl, prior_knowledge)


def _train_iter_from_config(config: DatasetConfig) -> enn_base.BatchIterator:
  """Return an iterator for train data of a real dataset specified by config."""
  train_iter, _ = load_dataset(
      name=config.dataset_name,
      split='train',
      batch_size=config.train_batch_size,
      is_training=True)
  return train_iter


def _eval_datasets_from_config(
    config: DatasetConfig) -> Dict[str, enn_base.BatchIterator]:
  """Return evaluation data as a dictionary of test/train iterators for real dataset specified by config."""
  eval_train_iter, _ = load_dataset(
      name=config.dataset_name,
      split='train',
      batch_size=config.eval_batch_size,
      is_training=False)
  eval_test_iter, _ = load_dataset(
      name=config.dataset_name,
      split='test',
      batch_size=config.eval_batch_size,
      is_training=False)
  eval_datasets = {'train': eval_train_iter, 'test': eval_test_iter}

  return eval_datasets


def problem_from_ds_name(config: DatasetConfig) -> RealDataProblem:
  """Load a classification problem from a real dataset specified by config."""
  testbed = _testbed_from_config(config)
  train_iter = _train_iter_from_config(config)
  if config.with_eval:
    eval_datasets = _eval_datasets_from_config(config)
  else:
    eval_datasets = None
  return testbed, train_iter, eval_datasets
