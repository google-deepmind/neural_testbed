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
"""Factory methods for Bayes by Backprop agent."""

import dataclasses
from typing import Sequence

from enn import base as enn_base
from enn import losses
from enn import networks

import jax.numpy as jnp

from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent

import optax


@dataclasses.dataclass
class BBBConfig:
  """Configuration for bbb agent."""
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for the neural network
  num_batches: int = 1000  # Number of SGD steps
  learning_rate: float = 3e-3  # Learning rate for adam optimizer
  seed: int = 0  # Initialization seed
  sigma_1: float = 1.0  # Standard deviation of the first Gaussian prior
  sigma_2: float = 0.75  # Standard deviation of the second Gaussian prior
  mixture_scale: float = 1.  # Scale for mixture of two Gauusian densities
  num_index_samples: int = 8  # Number of index samples to average over
  kl_method: str = 'analytical'  # How to find KL of prior and vi posterior
  adaptive_scale: bool = True  # Whether to scale prior KL with temp
  output_scale: bool = False  # Whether to scale output with temperature
  batch_strategy: bool = False  # Whether to scale num_batches with data ratio


def make_agent(config: BBBConfig) -> enn_agent.VanillaEnnAgent:
  """Factory method to create a BBB agent."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    """Makes ENN."""
    temperature = 1.
    if config.output_scale:
      temperature = prior.temperature
    enn = networks.make_bbb_enn(
        base_output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        dummy_input=jnp.zeros(shape=(prior.input_dim,)),
        temperature=temperature)

    return enn

  def make_loss(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
    """Define the ENN architecture from the prior."""
    del enn
    # Loss assuming a classification task.
    assert prior.num_classes > 1
    log_likelihood_fn = losses.get_categorical_loglike_fn(
        num_classes=prior.num_classes)
    if config.kl_method == 'analytical':
      model_prior_kl_fn = losses.get_analytical_diagonal_linear_model_prior_kl_fn(
          prior.num_train, config.sigma_1)
    elif config.kl_method == 'sample_based':
      model_prior_kl_fn = losses.get_sample_based_model_prior_kl_fn(
          prior.num_train, config.sigma_1, config.sigma_2,
          config.mixture_scale)
    else:
      ValueError(f'Invalid kl_method={config.kl_method}')

    if config.adaptive_scale:
      single_loss = losses.ElboLoss(log_likelihood_fn, model_prior_kl_fn,
                                    prior.temperature, prior.input_dim)
    else:
      single_loss = losses.ElboLoss(log_likelihood_fn, model_prior_kl_fn)

    loss_fn = losses.average_single_index_loss(
        single_loss, num_index_samples=config.num_index_samples)
    return loss_fn

  def batch_strategy(prior: testbed_base.PriorKnowledge) -> int:
    if not config.batch_strategy:
      return config.num_batches
    data_ratio = prior.num_train / prior.input_dim
    if data_ratio > 500:  # high data regime
      return config.num_batches * 5
    elif data_ratio < 5:  # low data regime
      return config.num_batches // 5
    else:
      return config.num_batches

  agent_config = enn_agent.VanillaEnnConfig(
      enn_ctor=make_enn,
      loss_ctor=make_loss,
      num_batches=batch_strategy,
      optimizer=optax.adam(config.learning_rate),
      seed=config.seed,
  )
  return enn_agent.VanillaEnnAgent(agent_config)
