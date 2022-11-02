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
"""Factory methods for ensemble_plus agent."""

import dataclasses
from typing import  Optional, Sequence

from enn import data_noise
from enn import losses
from enn import networks
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent
import numpy as np


@dataclasses.dataclass
class EnsembleConfig:
  """Config for ensemble with prior functions."""
  num_ensemble: int = 100  # Size of ensemble
  l2_weight_decay: float = 1.  # Weight decay
  adaptive_weight_scale: bool = True  # Whether to scale with prior
  distribution: str = 'none'  # Boostrap distribution
  prior_scale: float = 3.  # Scale of prior function
  temp_scale_prior: str = 'sqrt'  # How to scale prior with temperature
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for the neural network
  num_batches: int = 1_000  # Number of SGD steps
  batch_strategy: bool = False  # Whether to scale num_batches with data ratio
  seed: int = 0  # Initialization seed
  override_index_samples: Optional[int] = None  # Set SGD training index samples


def make_agent(config: EnsembleConfig) -> testbed_base.TestbedAgent:
  """Factory method to create a ensemble with prior."""

  num_index_samples = config.override_index_samples or config.num_ensemble

  def make_enn(prior: testbed_base.PriorKnowledge) -> networks.EnnArray:
    prior_scale = config.prior_scale
    if config.temp_scale_prior == 'linear':
      prior_scale /= prior.temperature
    elif config.temp_scale_prior == 'sqrt':
      prior_scale /= float(jnp.sqrt(prior.temperature))
    else:
      pass
    enn = networks.make_ensemble_mlp_with_prior_enn(
        output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        dummy_input=jnp.ones([100, prior.input_dim]),
        num_ensemble=config.num_ensemble,
        prior_scale=prior_scale,
        seed=config.seed + 999,
    )
    return enn

  def make_loss(prior: testbed_base.PriorKnowledge,
                enn: networks.EnnArray) -> losses.LossFnArray:
    """You can override this function to try different loss functions."""
    single_loss = losses.combine_single_index_losses_as_metric(
        train_loss=losses.XentLoss(prior.num_classes),
        extra_losses={
            'acc': losses.AccuracyErrorLoss(prior.num_classes)
        },
    )

    # Adding bootstrapping
    boot_fn = data_noise.BootstrapNoise(enn, config.distribution, config.seed)
    single_loss = losses.add_data_noise(single_loss, boot_fn)

    # Averaging over index
    loss_fn = losses.average_single_index_loss(single_loss,
                                               num_index_samples)

    # Adding weight decay
    scale = config.l2_weight_decay / config.num_ensemble
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
      num_batches=batch_strategy,
      seed=config.seed,)

  return enn_agent.VanillaEnnAgent(agent_config)
