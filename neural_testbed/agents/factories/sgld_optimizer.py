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
"""Optax implementations of SGMCMC optimizers.

Code is based on
https://github.com/google-research/google-research/tree/master/bnn_hmc
"""

from typing import Any, NamedTuple

import jax
from jax import numpy as jnp
from neural_testbed.agents.factories import preconditioner as pre
from optax import GradientTransformation
from optax import OptState
from optax import Params


Momentum = Any  # An arbitrary pytree of `jnp.ndarrays`
GradMomentEstimates = Params  # Same type as parameters
PreconditionerState = NamedTuple  # State of a preconditioner


def normal_like_tree(a, key):
  """Generate Gaussian noises."""
  treedef = jax.tree_structure(a)
  num_vars = len(jax.tree_leaves(a))
  all_keys = jax.random.split(key, num=(num_vars + 1))
  noise = jax.tree_multimap(lambda p, k: jax.random.normal(k, shape=p.shape), a,
                            jax.tree_unflatten(treedef, all_keys[1:]))
  return noise, all_keys[0]


class OptaxSGLDState(OptState):
  """Optax state for the SGLD optimizer."""
  count: jnp.ndarray
  rng_key: jnp.ndarray
  momentum: Momentum
  preconditioner_state: PreconditionerState


def sgld_gradient_update(step_size,
                         seed,
                         momentum_decay=0.,
                         preconditioner=None,
                         temperature=0.1):
  """Optax implementation of the SGLD optimizer.

  If momentum_decay is not zero,
  we get the underdamped SGLD (SGHMC) method "Stochastic Gradient Hamiltonian
        Monte Carlo" Tianqi Chen, Emily B. Fox, Carlos Guestrin; ICML 2014.

  Args:
    step_size: learning rate
    seed: int, random seed.
    momentum_decay: float, momentum decay parameter (default: 0).
    preconditioner: Preconditioner, an object representing the preconditioner
      or None; if None, identity preconditioner is used (default: None).
    temperature: algorithm temperature. If temperature<1, return a cold
      posterior.

  Returns:
    Optax.GradientTransformation.
  """

  if preconditioner is None:
    preconditioner = pre.get_identity_preconditioner()

  def init_fn(params):
    return OptaxSGLDState(
        count=jnp.zeros([], jnp.int32),
        rng_key=jax.random.PRNGKey(seed),
        momentum=jax.tree_map(jnp.zeros_like, params),
        preconditioner_state=preconditioner.init(params))

  def update_fn(gradient, state, params=None):
    del params
    noise_std = jnp.sqrt(2 * (1 - momentum_decay) * temperature)

    preconditioner_state = preconditioner.update_preconditioner(
        gradient, state.preconditioner_state)

    noise, new_key = normal_like_tree(gradient, state.rng_key)
    noise = preconditioner.multiply_by_m_sqrt(noise, preconditioner_state)

    def update_momentum(m, g, n):
      return momentum_decay * m + g * jnp.sqrt(step_size) - n * noise_std

    momentum = jax.tree_multimap(update_momentum, state.momentum, gradient,
                                 noise)
    updates = preconditioner.multiply_by_m_inv(momentum, preconditioner_state)
    updates = jax.tree_map(lambda m: -m * jnp.sqrt(step_size), updates)
    return updates, OptaxSGLDState(
        count=state.count + 1,
        rng_key=new_key,
        momentum=momentum,
        preconditioner_state=preconditioner_state)

  return GradientTransformation(init_fn, update_fn)
