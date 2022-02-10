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

"""Tests for neural_testbed.real_data.utils."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from neural_testbed.real_data import datasets
from neural_testbed.real_data import utils
import numpy as np
import tensorflow as tf


class LoadTest(parameterized.TestCase):

  @parameterized.parameters((np.array([-1., 0., 1.]),),
                            (np.array([0., 1., 2.]),))
  def test_standardize_data(self, array: np.ndarray):
    """Test data standardization method."""
    array = tf.convert_to_tensor(array, dtype=tf.float16)
    standardized_array = utils._standardize_data(array, epsilon=0)

    np.testing.assert_allclose(
        np.array(standardized_array),
        np.array([-1.225, 0., 1.225]),
        rtol=1e-3,
        atol=0)

  @parameterized.parameters((np.random.rand(2,), np.random.randint(10)), ({
      'f1': np.random.randint(100),
      'f2': np.random.randint(100)
  }, np.random.randint(10)))
  def test_preprocess_structured_data(self, features: utils.Features,
                                      label: int):
    """Test converting structured data into testbed standardized dictionary format."""
    data = utils._preprocess_structured_data(features, label)

    chex.assert_shape(data.x, (2,))
    chex.assert_shape(data.y, (1,))

  @parameterized.parameters(
      (np.random.rand(2, 2, 3), np.random.randint(10)),)
  def test_preprocess_image_data(self, image: chex.Array, label: int):
    """Test converting image data into testbed standardized dictionary format."""
    dataset = {'image': image, 'label': label}
    data = utils._preprocess_image_data(dataset)

    x_size = np.prod(image.shape)
    chex.assert_shape(data.x, (x_size,))
    chex.assert_shape(data.y, (1,))

  @parameterized.product(
      dataset_name=['iris', 'mnist'], split=['train', 'test'])
  def test_load_classification_dataset(self, dataset_name: str, split: str):
    """Test loading classification dataset."""
    dataset_info = datasets.DATASETS_SETTINGS[dataset_name]
    data = utils.load_classification_dataset(dataset_info, split)

    if split == 'train':
      data_size = dataset_info.num_train
    else:
      data_size = dataset_info.num_test

    chex.assert_shape(data.x, (data_size, dataset_info.input_dim))
    chex.assert_shape(data.y, (data_size, 1))

  @parameterized.product(dataset_name=['boston_housing', 'concrete_strength'])
  def test_load_regression_dataset(self, dataset_name: str):
    """Test loading regression dataset."""
    dataset_info = datasets.DATASETS_SETTINGS[dataset_name]
    train_data, test_data = utils.load_regression_dataset(dataset_name)

    # Check train data
    chex.assert_shape(train_data.x,
                      (dataset_info.num_train, dataset_info.input_dim))
    chex.assert_shape(train_data.y, (dataset_info.num_train, 1))

    # Check test data
    chex.assert_shape(
        test_data.x,
        (dataset_info.num_test, dataset_info.input_dim))
    chex.assert_shape(test_data.y, (dataset_info.num_test, 1))


if __name__ == '__main__':
  absltest.main()
