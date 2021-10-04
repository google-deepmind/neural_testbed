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
"""A fake data sampler for classification based on real data."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed import likelihood


class RealDataSampler(likelihood.GenerativeDataSampler):
  """A fake data sampler for classification/regression based on real data."""

  def __init__(self,
               train_data: testbed_base.Data,
               test_data: testbed_base.Data,
               tau: int = 1):
    self._train_data = train_data
    self._test_data = test_data
    self._tau = tau
    self._x_test = jnp.array(self._test_data.x)
    self._y_test = jnp.array(self._test_data.y)
    self._num_test_data = self._x_test.shape[0]

  @property
  def train_data(self) -> testbed_base.Data:
    """Returns train data."""
    return self._train_data

  def test_data(self, key: chex.PRNGKey) -> Tuple[testbed_base.Data, float]:
    """Returns a batch of test data.

    This method is called using self._num_test_data consecutive seeds.
    Args:
      key: A key for generating random numbers.

    Returns:
      A tuple of 1 test data as a testbed_base.Data and a float which is always
      0. The reason for having this float value equal to 0 is that in the
      testbed pipeline this method is expected to return a tuple of a test data
      and log-likelihood under posterior. However, for real data, we don't have
      log-likelihood under posterior. Hence, we set it to 0 and by doing so, we
      can still use one formula in the testbed for calculating kl estimate.
    """
    data_index = jax.random.randint(key, [self._tau], 0, self._num_test_data)
    x_test = self._x_test[data_index, :]
    y_test = self._y_test[data_index, :]
    return testbed_base.Data(x=x_test, y=y_test), 0.
