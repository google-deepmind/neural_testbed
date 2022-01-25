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
"""Factory methods for ensemble_plus agent."""

import dataclasses
from typing import Sequence

from enn import base as enn_base
from enn import data_noise
from enn import losses
from enn import networks
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent
from neural_testbed.agents.factories import base as factories_base
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
  batch_strategy: bool = True  # Whether to scale num_batches with data ratio
  seed: int = 0  # Initialization seed


def make_agent(config: EnsembleConfig) -> testbed_base.TestbedAgent:
  """Factory method to create a ensemble with prior."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    prior_scale = config.prior_scale
    if config.temp_scale_prior == 'linear':
      prior_scale /= prior.temperature
    elif config.temp_scale_prior == 'sqrt':
      prior_scale /= float(jnp.sqrt(prior.temperature))
    else:
      pass
    return networks.make_ensemble_mlp_with_prior_enn(
        output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        dummy_input=jnp.ones([100, prior.input_dim]),
        num_ensemble=config.num_ensemble,
        prior_scale=prior_scale,
        seed=config.seed + 999,
    )

  def make_loss(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
    """You can override this function to try different loss functions."""
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
    loss_fn = losses.average_single_index_loss(single_loss, config.num_ensemble)

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


def basic_sweep() -> Sequence[EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for num_ensemble in [1, 3, 10, 30, 100]:
    sweep.append(EnsembleConfig(
        num_ensemble=num_ensemble,
    ))
  return tuple(sweep)


def boot_sweep() -> Sequence[EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for distribution in ['none', 'bernoulli', 'exponential']:
    sweep.append(EnsembleConfig(
        distribution=distribution,
    ))
  return tuple(sweep)


def weight_decay_sweep() -> Sequence[EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for l2_weight_decay in [0.1, 0.3, 1, 3, 10]:
    sweep.append(EnsembleConfig(
        l2_weight_decay=l2_weight_decay,
    ))
  return tuple(sweep)


def prior_scale_sweep() -> Sequence[EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for temp_scale_prior in ['lin', 'sqrt']:
    for prior_scale in [1, 3]:
      sweep.append(EnsembleConfig(
          temp_scale_prior=temp_scale_prior,
          prior_scale=prior_scale
      ))
  return tuple(sweep)


def batch_sweep() -> Sequence[EnsembleConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for batch_strategy in [True, False]:
    for num_batches in [500, 1000]:
      sweep.append(
          EnsembleConfig(
              batch_strategy=batch_strategy,
              num_batches=num_batches))
  return tuple(sweep)


def combined_sweep() -> Sequence[EnsembleConfig]:
  return tuple(basic_sweep()) + tuple(boot_sweep()) + tuple(
      weight_decay_sweep()) + tuple(prior_scale_sweep()) + tuple(batch_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=EnsembleConfig(),
      ctor=make_agent,
      sweep=combined_sweep,
  )
