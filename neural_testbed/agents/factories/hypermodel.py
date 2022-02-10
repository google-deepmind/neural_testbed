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
"""Factory methods for hypermodel agent."""

import dataclasses
from typing import Sequence

from enn import base as enn_base
from enn import data_noise
from enn import losses
from enn import networks
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent
import numpy as np
import optax


@dataclasses.dataclass
class HypermodelConfig:
  """Configuration for hypermodel agent."""
  index_dim: int = 5  # Index dimension
  num_index_samples: int = 128  # Number of index samples to average over
  prior_scale: float = 1  # Scale for additive prior function
  l2_weight_decay: float = 1.  # Weight decay
  adaptive_weight_scale: bool = True  # Whether to scale with prior knowledge
  temp_scale_prior: str = 'sqrt'  # How to scale prior with temperature
  distribution: str = 'none'  # Bootsrapping distribution
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for the neural network
  prior_hidden_sizes: Sequence[int] = (10,)  # Hidden sizes for prior network
  num_batches: int = 2000  # Number of SGD steps
  batch_strategy: bool = False  # Whether to scale num_batches with data ratio
  learning_rate: float = 1e-3  # Learning rate for adam optimizer
  seed: int = 0  # Initialization seed
  scale: bool = False  # Whether to scale the params or not


def make_hypermodel_agent(
    config: HypermodelConfig) -> enn_agent.VanillaEnnAgent:
  """Factory method to create a hypermodel."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    prior_scale = config.prior_scale
    if config.temp_scale_prior == 'lin':
      prior_scale /= prior.temperature
    elif config.temp_scale_prior == 'sqrt':
      prior_scale /= float(jnp.sqrt(prior.temperature))
    else:
      pass
    return networks.MLPHypermodelPriorIndependentLayers(
        base_output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        prior_scale=prior_scale,
        dummy_input=jnp.ones([100, prior.input_dim]),
        indexer=networks.ScaledGaussianIndexer(config.index_dim),
        prior_base_output_sizes=list(config.prior_hidden_sizes) +
        [prior.num_classes],
        hyper_hidden_sizes=[],
        seed=config.seed,
        scale=config.scale,
    )

  def make_loss(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:

    single_loss = losses.combine_single_index_losses_as_metric(
        # This is the loss you are training on.
        train_loss=losses.XentLoss(prior.num_classes),
        # We will also log the accuracy in classification.
        extra_losses={'acc': losses.AccuracyErrorLoss(prior.num_classes)},
    )

    # Adding bootstrapping
    boot_fn = data_noise.BootstrapNoise(enn, config.distribution, config.seed)
    single_loss = losses.add_data_noise(single_loss, boot_fn)

    # Averaging over index
    loss_fn = losses.average_single_index_loss(single_loss,
                                               config.num_index_samples)

    # Adding weight decay
    scale = config.l2_weight_decay
    scale /= prior.num_train
    if config.adaptive_weight_scale:
      scale *= np.sqrt(prior.temperature) * prior.input_dim
    loss_fn = losses.add_l2_weight_decay(loss_fn, scale=scale)
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
      optimizer=optax.adam(config.learning_rate),
      num_batches=batch_strategy,
      seed=config.seed,)

  return enn_agent.VanillaEnnAgent(agent_config)
