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

"""Tests for neural_testbed.real_data.real_data_classification_test."""

from absl.testing import absltest
from absl.testing import parameterized
from neural_testbed import real_data
import numpy as np


class RealDataClassificationTest(parameterized.TestCase):

  @parameterized.product(
      data_size=[2, 10],
      input_dim=[1, 10])
  def test_data(self, data_size: int, input_dim: int):
    """Test returns for the train_data and test_data methods."""
    data = [{
        'x': np.random.rand(data_size, input_dim),
        'y': np.random.rand(data_size, 1)
    }]
    train_iter, test_iter = iter(data), iter(data)
    data_sampler = real_data.RealDataClassification(
        train_iter=train_iter, test_iter=test_iter)
    train_data = data_sampler.train_data
    np.testing.assert_allclose(train_data.x, data[0]['x'], rtol=1e-6, atol=0)
    np.testing.assert_allclose(train_data.y, data[0]['y'], rtol=1e-6, atol=0)

    for seed in range(data_size):
      test_data, ll = data_sampler.test_data(seed)
      self.assertEqual(ll, 0.)
      np.testing.assert_allclose(
          test_data.x,
          np.expand_dims(data[0]['x'][seed], axis=0),
          rtol=1e-6,
          atol=0)
      np.testing.assert_allclose(
          test_data.y,
          np.expand_dims(data[0]['y'][seed], axis=0),
          rtol=1e-6,
          atol=0)


if __name__ == '__main__':
  absltest.main()
