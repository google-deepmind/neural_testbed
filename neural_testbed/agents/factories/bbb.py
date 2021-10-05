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
"""Factory methods for Bayes by Backprop agent."""

import dataclasses
from typing import Sequence

from enn import base as enn_base
from enn import losses
from enn import networks

import jax.numpy as jnp

from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent
from neural_testbed.agents.factories import base as factories_base

import optax


@dataclasses.dataclass
class BBBConfig:
  """Configuration for bbb agent."""
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for the neural network
  num_batches: int = 10_000  # Number of SGD steps
  learning_rate: float = 3e-3  # Learning rate for adam optimizer
  seed: int = 0  # Initialization seed
  sigma_1: float = 0.3  # Standard deviation of the first Gaussian prior
  sigma_2: float = 0.75  # Standard deviation of the second Gaussian prior
  mixture_scale: float = 1.  # Scale for mixture of two Gauusian densities
  num_index_samples: int = 8  # Number of index samples to average over
  kl_method: str = 'analytical'  # How to find KL of prior and vi posterior
  adaptive_scale: bool = True  # Whether to scale prior KL with temp
  output_scale: bool = False  # Whether to scale output with temperature


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
                                    prior.temperature)
    else:
      single_loss = losses.ElboLoss(log_likelihood_fn, model_prior_kl_fn)

    loss_fn = losses.average_single_index_loss(
        single_loss, num_index_samples=config.num_index_samples)
    return loss_fn

  agent_config = enn_agent.VanillaEnnConfig(
      enn_ctor=make_enn,
      loss_ctor=make_loss,
      num_batches=config.num_batches,
      optimizer=optax.adam(config.learning_rate),
      seed=config.seed,
  )
  return enn_agent.VanillaEnnAgent(agent_config)


def base_sweep() -> Sequence[BBBConfig]:
  """Generates the bbb sweep over network parameters."""
  sweep = []
  for learning_rate in [1e-3, 3e-3]:
    for sigma_1 in [0.3, 0.5, 0.7, 1, 2]:
      for num_batches in [500, 1000, 10_000]:
        sweep.append(
            BBBConfig(
                learning_rate=learning_rate,
                sigma_1=sigma_1,
                num_batches=num_batches))
  return tuple(sweep)


def prior_sweep() -> Sequence[BBBConfig]:
  """Generates the bbb sweep over prior parameters."""
  sweep = []
  for sigma_1 in [1, 2, 4]:
    for sigma_2 in [0.25, 0.5, 0.75]:
      for mixture_scale in [0, 0.5, 1]:
        for num_batches in [1000, 10_000]:
          sweep.append(BBBConfig(sigma_1=sigma_1,
                                 sigma_2=sigma_2,
                                 mixture_scale=mixture_scale,
                                 num_batches=num_batches))
  return tuple(sweep)


def network_sweep() -> Sequence[BBBConfig]:
  """Generates the bbb sweep over network architecture for paper."""
  sweep = []
  for hidden_sizes in [(50, 50), (100, 100), (50, 50, 50)]:
    sweep.append(BBBConfig(hidden_sizes=hidden_sizes))
  return tuple(sweep)


def combined_sweep() -> Sequence[BBBConfig]:
  return tuple(base_sweep()) + tuple(prior_sweep()) + tuple(network_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=BBBConfig(),
      ctor=make_agent,
      sweep=combined_sweep,
  )
