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
"""Factory methods for ensemble agent."""

import dataclasses
from typing import Sequence

from enn import base as enn_base
from enn import losses
from enn import networks
from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent
import numpy as np


@dataclasses.dataclass
class VanillaEnsembleConfig:
  num_ensemble: int = 100  # Size of ensemble
  l2_weight_decay: float = 1.  # Weight decay
  adaptive_weight_scale: bool = True  # Whether to scale with prior
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for the neural network
  num_batches: int = 1000  # Number of SGD steps
  batch_strategy: bool = False  # Whether to scale num_batches with data ratio
  seed: int = 0  # Initialization seed


def make_agent(config: VanillaEnsembleConfig) -> enn_agent.VanillaEnnAgent:
  """Factory method to create a vanilla ensemble."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    return networks.make_einsum_ensemble_mlp_enn(
        output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        num_ensemble=config.num_ensemble,
        nonzero_bias=False,
    )

  def make_loss(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
    del enn
    single_loss = losses.combine_single_index_losses_as_metric(
        # This is the loss you are training on.
        train_loss=losses.XentLoss(prior.num_classes),
        # We will also log the accuracy in classification.
        extra_losses={'acc': losses.AccuracyErrorLoss(prior.num_classes)},
    )

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
      seed=config.seed,
  )
  return enn_agent.VanillaEnnAgent(agent_config)
