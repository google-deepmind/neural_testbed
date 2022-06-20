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
"""Code to adjust testbed agents for sequential decision problems.

The focus of this code is to add functionality for a sensible decay of "prior
effect" as the number of training points grow.
"""
import typing
from typing import Tuple

import chex
from enn import base as enn_base
from enn import losses
from enn import networks
import haiku as hk
import jax.numpy as jnp
from neural_testbed import agents
from neural_testbed import base as testbed_base
from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import bbb


def make_config_l2_for_bandit(
    paper_agent: factories_base.PaperAgent,
    temperature: float = 1,
    seed: int = 0,
) -> Tuple[agents.VanillaEnnConfig, float]:
  """Converts agent to sequential decision form and appropriate l2 decay."""
  # Extract agent config
  config = paper_agent.default

  # Extract the l2 weight decay parameter from the agent default.
  # Then, override that paper.default to be zero so we don't double l2 decay.
  if hasattr(config, 'l2_weight_decay'):
    l2_weight_decay = config.l2_weight_decay
    config.l2_weight_decay = 0
  elif hasattr(config, 'dropout_rate'):
    l2_weight_decay = config.length_scale
    config.length_scale = 0
  else:
    l2_weight_decay = 0

  # Rescale l2 weight decay by temperature, and potentially by ensemble size
  l2_weight_decay *= 2 * temperature
  if hasattr(config, 'num_ensemble'):
    l2_weight_decay = l2_weight_decay / config.num_ensemble

  # Override seed and form agent
  config.seed = seed
  agent = paper_agent.ctor(config)
  assert isinstance(agent, agents.VanillaEnnAgent)
  agent = typing.cast(agents.VanillaEnnAgent, agent)

  # If the agent is bbb then we should override the loss_fn
  if isinstance(config, bbb.BBBConfig):
    agent.config.loss_ctor = _make_bbb_bandit_loss(config)

  return agent.config, l2_weight_decay


def _make_bbb_bandit_loss(config: bbb.BBBConfig) -> agents.LossCtor:
  """BBB loss with decaying prior through time for sequential decisions."""

  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: networks.EnnNoState) -> losses.LossFnNoState:
    del enn
    log_likelihood_fn = losses.get_categorical_loglike_fn(prior.num_classes)
    prior_kl_fn = losses.get_analytical_diagonal_linear_model_prior_kl_fn(
        1, config.sigma_1)

    def elbo_loss(
        apply: networks.ApplyNoState,
        params: hk.Params,
        batch: enn_base.Batch,
        index: enn_base.Index,
    ) -> Tuple[chex.Array, enn_base.LossMetrics]:
      """Elbo loss with decay per num_steps stored in the batch."""
      out = apply(params, batch.x, index)
      log_likelihood = log_likelihood_fn(out, batch)
      prior_kl = prior_kl_fn(out, params, index)
      chex.assert_equal_shape([log_likelihood, prior_kl])
      # Rescaling by num_steps and temperature
      prior_kl *= 2 * jnp.sqrt(prior.temperature) / batch.extra['num_steps']
      return prior_kl - log_likelihood, {}

    return losses.average_single_index_loss(elbo_loss, config.num_index_samples)
  return loss_ctor
