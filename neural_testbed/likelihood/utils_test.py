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

"""Tests for neural_testbed.likelihood."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from neural_testbed.likelihood import utils


class LogSumProdTest(parameterized.TestCase):

  @parameterized.product(
      num_centroids=[10, 100],
      tau=[100, 1000],
      magnitude=[-12, -10, -8],
  )
  def test_not_nan(self, num_centroids: int, tau: int, magnitude: float):
    """Check that we don't get Inf."""

    def compute_ll(key: chex.PRNGKey) -> float:
      num_obs = jax.random.poisson(key, 1, [num_centroids])
      nu = num_obs / jnp.sum(num_obs)

      q_hat = jnp.ones([num_centroids, tau]) * (10 ** magnitude)
      q_hat += jnp.expand_dims(nu == 0, 1).astype(jnp.float32)
      q_hat = jnp.clip(q_hat, 0, 1)

      return utils.log_sum_prod(nu, q_hat)

    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    log_likelihoods = jax.jit(jax.vmap(compute_ll))(keys)
    assert jnp.all(jnp.isfinite(log_likelihoods))


if __name__ == '__main__':
  absltest.main()
