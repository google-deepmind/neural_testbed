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

"""Loading a classification real data problem for the testbed."""

from typing import Dict, Union

import chex
import haiku as hk
from neural_testbed import base as testbed_base
from neural_testbed import likelihood
from neural_testbed.real_data import data_sampler as real_data
from neural_testbed.real_data import datasets
import tensorflow as tf
import tensorflow_datasets as tfds

Features = Union[Dict[str, Union[int, float]], chex.Array]


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


def load_dataset(
    dataset_info: datasets.DatasetInfo, split: str) -> testbed_base.Data:
  """Loads the dataset as an iterator of batches."""
  dataset_name = dataset_info.dataset_name
  if dataset_name not in datasets.CLASSIFICATION_DATASETS:
    raise ValueError(f'dataset {dataset_name} is not supported yet.')
  if dataset_name in datasets.STRUCTURED_DATASETS:
    return _load_structured_dataset(dataset_info=dataset_info, split=split)
  else:
    return _load_image_dataset(dataset_info=dataset_info, split=split)


def load(dataset_info: datasets.DatasetInfo,
         tau: int = 1) -> likelihood.SampleBasedTestbed:
  """Load a classification problem from a real dataset specified by config."""
  rng = hk.PRNGSequence(999)
  num_enn_samples = 1000  # We set it to the number we use for our testbed
  num_test_seeds = int(max(1000 / tau, 1))  # Match testbed
  train_data = load_dataset(dataset_info=dataset_info, split='train')
  test_data = load_dataset(dataset_info=dataset_info, split='test')

  data_sampler = real_data.RealDataSampler(train_data, test_data, tau)

  # TODO(lxlu): Tau threshold may need to depend on number of classes.
  if tau >= 10:
    sample_based_kl = likelihood.CategoricalClusterKL(
        cluster_alg=likelihood.RandomProjection(dimension=10),
        num_enn_samples=num_enn_samples,
        num_test_seeds=num_test_seeds,
        key=next(rng),
    )
  else:
    sample_based_kl = likelihood.CategoricalKLSampledXSampledY(
        num_test_seeds=num_test_seeds,
        num_enn_samples=num_enn_samples,
        # TODO(author1): Verify that fixed rngk is not causing issues.
        key=next(rng),
        num_classes=dataset_info.num_classes,
    )
  sample_based_kl = likelihood.add_classification_accuracy_ece(
      sample_based_kl,
      num_test_seeds=int(1_000 / tau) + 1,
      num_enn_samples=100,
      num_classes=dataset_info.num_classes,
  )
  prior_knowledge = testbed_base.PriorKnowledge(
      input_dim=dataset_info.input_dim,
      num_train=dataset_info.num_train,
      num_classes=dataset_info.num_classes,
      temperature=0.01,
      tau=tau,
  )
  return likelihood.SampleBasedTestbed(
      data_sampler, sample_based_kl, prior_knowledge)
