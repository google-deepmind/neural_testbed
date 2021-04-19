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

"""Tests for neural_testbed.real_data.load."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from neural_testbed.real_data import load
import numpy as np
import tensorflow as tf


class LoadTest(parameterized.TestCase):

  @parameterized.parameters((np.array([-1., 0., 1.]),),
                            (np.array([0., 1., 2.]),))
  def test_standardize_data(self, array: np.ndarray):
    """Test data standardization method."""
    array = tf.convert_to_tensor(array, dtype=tf.float16)
    standardized_array = load._standardize_data(array, epsilon=0)

    np.testing.assert_allclose(
        np.array(standardized_array),
        np.array([-1.225, 0., 1.225]),
        rtol=1e-3,
        atol=0)

  @parameterized.parameters(
      (np.random.randint(100), np.random.rand(2,), np.random.randint(10)),
      (np.random.randint(100), {
          'f1': np.random.randint(100),
          'f2': np.random.randint(100)
      }, np.random.randint(10)))
  def test_preprocess_structured_data(self, data_index: int,
                                      features: load.Features, label: int):
    """Test converting structured data into testbed standardized dictionary format."""
    data_dict = load._preprocess_structured_data(data_index, (features, label))
    self.assertIn('x', data_dict)
    self.assertIn('y', data_dict)
    self.assertIn('label', data_dict)
    self.assertIn('data_index', data_dict)

    chex.assert_shape(data_dict['x'], (2,))
    chex.assert_shape(data_dict['y'], (1,))
    chex.assert_shape(data_dict['label'], (1,))
    chex.assert_shape(data_dict['data_index'], (1,))

  @parameterized.parameters(
      (np.random.randint(100), np.random.rand(2, 2, 3), np.random.randint(10)),)
  def test_preprocess_image_data(self, data_index: int, image: chex.Array,
                                 label: int):
    """Test converting image data into testbed standardized dictionary format."""
    dataset = {'image': image, 'label': label}
    data_dict = load._preprocess_image_data(data_index, dataset)
    self.assertIn('x', data_dict)
    self.assertIn('y', data_dict)
    self.assertIn('label', data_dict)
    self.assertIn('data_index', data_dict)

    x_size = np.prod(image.shape)
    chex.assert_shape(data_dict['x'], (x_size,))
    chex.assert_shape(data_dict['y'], (1,))
    chex.assert_shape(data_dict['label'], (1,))
    chex.assert_shape(data_dict['data_index'], (1,))

  @parameterized.product(
      name=['iris', 'cmaterdb', 'wine_quality'],
      split=['train', 'test'],
      batch_size=[1, 10],
      is_training=[True, False])
  def test_load_dataset(self, name: str, split: str, batch_size: int,
                        is_training: bool):
    """Test loading dataset."""
    data_iter, info = load.load_dataset(
        name=name, split=split, batch_size=batch_size, is_training=is_training)
    self.assertIsInstance(info, load.DatasetInfo)

    data_dict = next(data_iter)
    self.assertIn('x', data_dict)
    self.assertIn('y', data_dict)
    self.assertIn('label', data_dict)
    self.assertIn('data_index', data_dict)

    chex.assert_shape(data_dict['x'], (batch_size, info.input_dim))
    chex.assert_shape(data_dict['y'], (batch_size, 1))
    chex.assert_shape(data_dict['label'], (batch_size, 1))
    chex.assert_shape(data_dict['data_index'], (batch_size, 1))


if __name__ == '__main__':
  absltest.main()
