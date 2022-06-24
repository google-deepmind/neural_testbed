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
"""Factory methods for epinet agent."""

import dataclasses
from typing import Sequence

from enn import data_noise
from enn import losses
from enn import networks
from enn.networks import epinet
import jax.numpy as jnp
from neural_testbed import agents
from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent


@dataclasses.dataclass
class EpinetConfig:
  """Config for ensemble with prior functions."""
  index_dim: int = 8  # Index dimension
  l2_weight_decay: float = 0.2  # Weight decay
  distribution: str = 'none'  # Bootstrap distribution
  prior_scale: float = 0.3  # Scale of the additive prior function
  prior_scale_epi: float = 0.  # Scale of the epinet prior function
  prior_loss_freq: int = 100_000  # Prior loss frequency
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for the neural network
  num_batches: int = 1_000  # Number of SGD steps
  epi_hiddens: Sequence[int] = (15, 15)  # Hidden sizes in epinet
  add_hiddens: Sequence[int] = (5, 5)  # Hidden sizes in additive prior
  seed: int = 0  # Initialization seed


def make_agent(config: EpinetConfig) -> enn_agent.VanillaEnnAgent:
  """Factory method to create an epinet agent with ensemble prior."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> networks.EnnArray:
    prior_scale = config.prior_scale / prior.temperature

    # We only want to expose final hidden layer, so we set the flag for previous
    # hidden layer and final output layer to False and for the final hidden
    # layer to True.
    expose_layers = [False,] * (len(config.hidden_sizes) - 1) + [True, False]
    enn = epinet.make_mlp_epinet(
        output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        epinet_hiddens=config.epi_hiddens,
        index_dim=config.index_dim,
        expose_layers=expose_layers,
        prior_scale=config.prior_scale_epi,
    )

    # Adding a linear combination of networks as prior function
    mlp_prior_fns = networks.make_mlp_ensemble_prior_fns(
        output_sizes=list(config.add_hiddens) + [prior.num_classes,],
        dummy_input=jnp.ones([100, prior.input_dim]),
        num_ensemble=config.index_dim,
        seed=config.seed,
    )
    mlp_prior_fn = networks.combine_functions_linear_in_index(mlp_prior_fns)
    return networks.EnnStateWithAdditivePrior(enn, mlp_prior_fn, prior_scale)

  def make_loss(prior: testbed_base.PriorKnowledge,
                enn: networks.EnnArray) -> losses.LossFnArray:
    """You can override this function to try different loss functions."""
    single_loss = losses.combine_single_index_losses_as_metric(
        train_loss=losses.XentLossWithState(prior.num_classes),
        extra_losses={
            'acc': losses.AccuracyErrorLossWithState(prior.num_classes)
        },
    )

    # Adding bootstrapping
    boot_fn = data_noise.BootstrapNoise(enn, config.distribution, config.seed)
    single_loss = losses.add_data_noise(single_loss, boot_fn)

    # Averaging over index
    loss_fn = losses.average_single_index_loss(single_loss, config.index_dim)

    # Adding weight decay
    scale = config.l2_weight_decay
    scale *= (prior.input_dim / prior.num_train) ** 0.7

    def predicate(module_name: str, name: str, value) -> bool:
      del name, value
      return 'prior' not in module_name
    loss_fn = losses.add_l2_weight_decay(loss_fn, scale, predicate)
    return loss_fn

  def num_batches(prior: testbed_base.PriorKnowledge) -> int:
    if (prior.num_train / prior.input_dim) > 500:
      return config.num_batches * 10
    else:
      return config.num_batches

  agent_config = enn_agent.VanillaEnnConfig(
      enn_ctor=make_enn,
      loss_ctor=make_loss,
      num_batches=num_batches,
      prior_loss_ctor=agents.default_enn_prior_loss(config.index_dim),
      prior_loss_freq=config.prior_loss_freq,
      seed=config.seed,
  )
  return enn_agent.VanillaEnnAgent(agent_config)
