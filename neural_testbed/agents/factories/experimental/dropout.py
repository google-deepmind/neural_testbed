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
"""Factory methods for MC Dropout agent."""

import dataclasses
from typing import Sequence

from enn import base as enn_base
from enn import losses
from enn import networks
from neural_testbed import agents
from neural_testbed import base as testbed_base
from neural_testbed.agents.factories import base as factories_base
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
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for neural network
  num_batches: int = 1000  # Number of SGD steps
  batch_strategy: bool = True  # Whether to scale num_batches with data ratio
  learning_rate: float = 1e-3  # Learning rate for adam optimizer
  seed: int = 0  # Initialization seed


def make_mc_dropout_agent(
    config: McDropoutConfig) -> agents.VanillaEnnAgent:
  """Factory method to create MC dropout agent."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    return networks.MLPDropoutENN(
        output_sizes=list(config.hidden_sizes) + [prior.num_classes],
        dropout_rate=config.dropout_rate,
        dropout_input=config.dropout_input,
        seed=config.seed,
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


def droprate_sweep() -> Sequence[McDropoutConfig]:
  """Generates the dropout sweep over dropping parameters for paper."""
  sweep = []
  for dropout_rate in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for dropout_input in [True, False]:
      sweep.append(
          McDropoutConfig(
              dropout_rate=dropout_rate,
              dropout_input=dropout_input))
  return tuple(sweep)


def l2reg_sweep() -> Sequence[McDropoutConfig]:
  """Generates the dropout sweep over l2 regularization parameters for paper."""
  sweep = []
  for adaptive_weight_scale in [True, False]:
    for length_scale in [0.01, 0.1, 0.3, 1, 3, 10]:
      sweep.append(
          McDropoutConfig(
              adaptive_weight_scale=adaptive_weight_scale,
              length_scale=length_scale))
  return tuple(sweep)


def network_sweep() -> Sequence[McDropoutConfig]:
  """Generates the dropout sweep over dropping parameters for paper."""
  sweep = []
  for hidden_sizes in [(50, 50), (100, 100)]:
    sweep.append(McDropoutConfig(hidden_sizes=hidden_sizes))
  return tuple(sweep)


def batch_sweep() -> Sequence[McDropoutConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for batch_strategy in [True, False]:
    for num_batches in [500, 1000]:
      sweep.append(
          McDropoutConfig(
              batch_strategy=batch_strategy,
              num_batches=num_batches))
  return tuple(sweep)


def combined_sweep() -> Sequence[McDropoutConfig]:
  return tuple(droprate_sweep()) + tuple(l2reg_sweep()) + tuple(
      network_sweep()) + tuple(batch_sweep())


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=McDropoutConfig(),
      ctor=make_mc_dropout_agent,
      sweep=combined_sweep,
      )
