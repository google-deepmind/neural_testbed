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

"""Tests for neural_testbed.real_data.data_sampler."""

from absl.testing import absltest
from absl.testing import parameterized
from neural_testbed import base as testbed_base
from neural_testbed.real_data import data_sampler as real_data_sampler
import numpy as np


class RealDataClassificationTest(parameterized.TestCase):

  @parameterized.product(
      data_size=[1, 10],
      input_dim=[1, 10],
      num_classes=[1, 2, 10],
      tau=[1, 10])
  def test_data(self,
                data_size: int,
                input_dim: int,
                num_classes: int,
                tau: int):
    """Test returns for the train_data and test_data methods."""
    x = np.random.rand(data_size, input_dim)
    y = np.random.randint(num_classes, size=(data_size, 1))
    data = testbed_base.Data(x=x, y=y)
    train_data, test_data = data, data
    data_sampler = real_data_sampler.RealDataSampler(
        train_data=train_data,
        test_sampler=real_data_sampler.make_global_sampler(test_data),
        tau=tau,
    )
    train_data = data_sampler.train_data
    np.testing.assert_allclose(train_data.x, x, rtol=1e-6, atol=0)
    np.testing.assert_allclose(train_data.y, y, rtol=1e-6, atol=0)


if __name__ == '__main__':
  absltest.main()
