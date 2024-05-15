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
"""Preconditioner for SGMCMC optimizers.

Code is based on
https://github.com/google-research/google-research/tree/master/bnn_hmc
"""

import dataclasses
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp
from optax import Params


Momentum = Any  # An arbitrary pytree of `jnp.ndarrays`
GradMomentEstimates = Params  # Same type as parameters
PreconditionerState = NamedTuple  # State of a preconditioner


@dataclasses.dataclass
class Preconditioner:
  """Preconditioner transformation."""
  init: Any
  update_preconditioner: Any
  multiply_by_m_sqrt: Any
  multiply_by_m_inv: Any
  multiply_by_m_sqrt_inv: Any


class RMSPropPreconditionerState(PreconditionerState):
  grad_moment_estimates: GradMomentEstimates


def get_rmsprop_preconditioner(running_average_factor=0.99, eps=1e-7):
  """Define RMSProp Preconditioner."""

  def init_fn(params):
    return RMSPropPreconditionerState(
        grad_moment_estimates=jax.tree.map(jnp.zeros_like, params)
    )

  def update_preconditioner_fn(gradient, preconditioner_state):
    r = running_average_factor
    grad_moment_estimates = jax.tree.map(
        lambda e, g: e * r + g**2 * (1 - r),
        preconditioner_state.grad_moment_estimates,
        gradient,
    )
    return RMSPropPreconditionerState(
        grad_moment_estimates=grad_moment_estimates)

  def multiply_by_m_inv_fn(vec, preconditioner_state):
    return jax.tree.map(
        lambda e, v: v / (eps + jnp.sqrt(e)),
        preconditioner_state.grad_moment_estimates,
        vec,
    )

  def multiply_by_m_sqrt_fn(vec, preconditioner_state):
    return jax.tree.map(
        lambda e, v: v * jnp.sqrt(eps + jnp.sqrt(e)),
        preconditioner_state.grad_moment_estimates,
        vec,
    )

  def multiply_by_m_sqrt_inv_fn(vec, preconditioner_state):
    return jax.tree.map(
        lambda e, v: v / jnp.sqrt(eps + jnp.sqrt(e)),
        preconditioner_state.grad_moment_estimates,
        vec,
    )

  return Preconditioner(
      init=init_fn,
      update_preconditioner=update_preconditioner_fn,
      multiply_by_m_inv=multiply_by_m_inv_fn,
      multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
      multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn)


class IdentityPreconditionerState(PreconditionerState):
  """Identity preconditioner is stateless."""


def get_identity_preconditioner():
  """Define Identity Preconditioner."""

  def init_fn(_):
    return IdentityPreconditionerState()

  def update_preconditioner_fn(gradient, preconditioner_state):
    del gradient, preconditioner_state
    return IdentityPreconditionerState()

  def multiply_by_m_inv_fn(vec, _):
    return vec

  def multiply_by_m_sqrt_fn(vec, _):
    return vec

  def multiply_by_m_sqrt_inv_fn(vec, _):
    return vec

  return Preconditioner(
      init=init_fn,
      update_preconditioner=update_preconditioner_fn,
      multiply_by_m_inv=multiply_by_m_inv_fn,
      multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
      multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn)
