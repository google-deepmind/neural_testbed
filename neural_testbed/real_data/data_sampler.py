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

from typing import Tuple, Callable

import chex
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed import likelihood

# key, num_samples -> Data
Sampler = Callable[[chex.PRNGKey, int], testbed_base.Data]
SamplerCtor = Callable[[testbed_base.Data], Sampler]


class RealDataSampler(likelihood.GenerativeDataSampler):
  """A fake data sampler for classification/regression based on real data."""

  def __init__(self,
               train_data: testbed_base.Data,
               test_sampler: Sampler,
               tau: int = 1):
    self._train_data = train_data
    self._tau = tau
    self._test_sampler = test_sampler

  @property
  def train_data(self) -> testbed_base.Data:
    """Returns train data."""
    return self._train_data

  def test_data(self, key: chex.PRNGKey) -> Tuple[testbed_base.Data, float]:
    """Returns a batch of test data.

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
    test_data = self._test_sampler(key, self._tau)
    return test_data, 0.


def make_local_sampler(data: testbed_base.Data, kappa: int = 2) -> Sampler:
  """Returns a sampler which samples based on kappa anchor points.

  To make this work in jax we actually implement this by first sampling kappa
  anchor points, then randomly the tau batch points from these kappa anchors
  (with replacement).

  Args:
    data: test data.
    kappa: number of anchor reference points. If tau is less than kappa we
      default to sampling tau points.

  Returns:
    Local sampler of data indices.
  """
  x_test = jnp.array(data.x)
  y_test = jnp.array(data.y)
  num_data = y_test.shape[0]

  def local_sampler(key: chex.PRNGKey, tau: int) -> testbed_base.Data:
    anchor_key, sample_key = jax.random.split(key, 2)
    # Sample anchor data indices
    anchor_idx = jax.random.randint(anchor_key, [kappa], 0, num_data)

    # Index into these anchor indices
    sample_idx = jax.random.randint(sample_key, [tau], 0, kappa)
    repeat_idx = anchor_idx[sample_idx]
    chex.assert_shape(repeat_idx, [tau])

    return testbed_base.Data(x=x_test[repeat_idx, :], y=y_test[repeat_idx, :])

  return local_sampler


def make_global_sampler(data: testbed_base.Data)-> Sampler:
  """Returns a sampler which samples uniformly from data points."""
  x_test = jnp.array(data.x)
  y_test = jnp.array(data.y)
  num_data = y_test.shape[0]

  def global_sampler(key: chex.PRNGKey, tau: int) -> testbed_base.Data:
    sample_idx = jax.random.randint(key, [tau], 0, num_data)
    chex.assert_shape(sample_idx, [tau])

    return testbed_base.Data(x=x_test[sample_idx, :], y=y_test[sample_idx, :])

  return global_sampler
