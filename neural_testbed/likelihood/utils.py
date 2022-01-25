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
"""Utility functions for likelihood code."""

import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from neural_testbed import base as testbed_base


def enumerate_all_features(dim: int, num_values: int) -> chex.Array:
  """Helper function to create all categorical features."""
  features = jnp.array(list(itertools.product(range(num_values), repeat=dim)))
  chex.assert_shape(features, [num_values ** dim, dim])
  return features.astype(jnp.int32)


def log_sum_prod(nu: chex.Array, q_hat: chex.Array) -> float:
  """Efficiently computes log(sum_k nu_k prod_t q_{k,t})."""
  chex.assert_rank([nu, q_hat], [1, 2])
  chex.assert_equal(nu.shape[0], q_hat.shape[0])

  # We have a numerical problem where the q_hat where nu are zero can be 0 or 1.
  # Since these terms ultimately do not contribute anything to the sum, we can
  # replace them in with anything. For numerical stability, we just set them to
  # be 1e-9.
  zero_nus = jnp.expand_dims(nu == 0, axis=1)
  amended_q_hat = q_hat * (1 - zero_nus) + (1e-9 * zero_nus)
  log_prod_qs = jnp.sum(jnp.log(amended_q_hat), axis=1)
  chex.assert_equal_shape([nu, log_prod_qs])

  # Use rebasing trick to maintain numerical stability
  base = jnp.max(log_prod_qs)
  ll = jsp.logsumexp(log_prod_qs - base, b=nu) + base
  return ll


def parse_kl_estimates(kl_estimates: chex.Array) -> testbed_base.ENNQuality:
  """Parse the finite elements of KL estimates."""
  # TODO(zhengwen): This section of the code is designed to clip errant inf.
  # We don't exactly know why this is happening but decide to clip infinite
  # estimate by the maximum finite KL and record the number of inf estimates.
  kl_estimates_finite = kl_estimates[jnp.isfinite(kl_estimates)]
  pct_finite_kl = len(kl_estimates_finite) / len(kl_estimates)
  kl_estimate_max = jax.lax.cond(
      len(kl_estimates_finite) >= 1,
      lambda _: jnp.max(_, initial=-jnp.inf),
      lambda _: jnp.inf,
      kl_estimates_finite,
  )
  clipped_estimates = jnp.minimum(kl_estimates, kl_estimate_max)
  kl_estimate = jnp.mean(clipped_estimates)
  extra = {
      'kl_estimate_std': float(jnp.std(kl_estimates)),
      'train_acc': None,
      'test_acc': None,
      'train_ece': None,
      'test_ece': None,
      'pct_finite_kl': float(pct_finite_kl)
  }
  return testbed_base.ENNQuality(kl_estimate=kl_estimate, extra=extra)

