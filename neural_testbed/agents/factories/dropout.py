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
"""Factory methods for MC Dropout agent."""

import dataclasses
from typing import Sequence

from enn import losses
from enn import networks
from neural_testbed import agents
from neural_testbed import base as testbed_base
import numpy as np
import optax


@dataclasses.dataclass
class McDropoutConfig:
  """Configuration for mc dropout agent."""
  dropout_rate: float = 0.1  # Drop probability for each hidden unit
  length_scale: float = 1.  # Length scale used for weight regularization
  regularization_tau: float = 1.  # tau for scaling the weight regularizer
  dropout_input: bool = False  # Whether to have dropout for the input layer
  exclude_bias_l2: bool = False  # Whether to exclude bias from regularization
  adaptive_weight_scale: bool = True  # Whether to scale with prior
  hidden_sizes: Sequence[int] = (100, 100)  # Hidden sizes for neural network
  num_batches: int = 1000  # Number of SGD steps
  batch_strategy: bool = False  # Whether to scale num_batches with data ratio
  learning_rate: float = 1e-3  # Learning rate for adam optimizer
  seed: int = 0  # Initialization seed


def make_mc_dropout_agent(
    config: McDropoutConfig) -> agents.VanillaEnnAgent:
  """Factory method to create MC dropout agent."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> networks.EnnArray:
    return networks.MLPDropoutENN(
        output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        dropout_rate=config.dropout_rate,
        dropout_input=config.dropout_input,
        seed=config.seed,
    )

  def make_loss(prior: testbed_base.PriorKnowledge,
                enn: networks.EnnArray) -> losses.LossFnArray:
    del enn
    single_loss = losses.combine_single_index_losses_as_metric(
        train_loss=losses.XentLossWithState(prior.num_classes),
        extra_losses={
            'acc': losses.AccuracyErrorLossWithState(prior.num_classes)
        },
    )

    # Averaging over index
    loss_fn = losses.average_single_index_loss(single_loss, num_index_samples=1)

    # Adding a special weight regularization based on paper "Dropout as a
    # Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
    # https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py#L72
    scale = (config.length_scale**2) * (1 - config.dropout_rate) / (
        2. * prior.num_train * config.regularization_tau)
    if config.adaptive_weight_scale:
      scale = config.length_scale * np.sqrt(
          prior.temperature) * prior.input_dim / prior.num_train
    if config.exclude_bias_l2:
      predicate = lambda module, name, value: name != 'b'
    else:
      predicate = lambda module, name, value: True
    loss_fn = losses.add_l2_weight_decay(loss_fn, scale, predicate)
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

  agent_config = agents.VanillaEnnConfig(
      enn_ctor=make_enn,
      loss_ctor=make_loss,
      optimizer=optax.adam(config.learning_rate),
      num_batches=batch_strategy,
      seed=config.seed,
  )
  return agents.VanillaEnnAgent(agent_config)
