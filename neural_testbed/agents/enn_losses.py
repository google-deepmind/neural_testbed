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

"""Helpful losses for the ENN agent."""

from typing import Callable, Optional

from enn import data_noise
from enn import losses
from enn import networks
from neural_testbed import base as testbed_base


EnnCtor = Callable[[testbed_base.PriorKnowledge], networks.EnnNoState]
LossCtor = Callable[
    [testbed_base.PriorKnowledge, networks.EnnNoState], losses.LossFnNoState]


def default_enn_prior_loss(num_index_samples: int = 10) -> LossCtor:
  def prior_loss_ctor(prior: testbed_base.PriorKnowledge,
                      enn: networks.EnnNoState) -> losses.LossFnNoState:
    del enn
    if prior.num_classes > 1:
      return losses.ClassificationPriorLoss(num_index_samples)
    else:
      return losses.RegressionPriorLoss(num_index_samples)
  return prior_loss_ctor


def default_enn_loss(num_index_samples: int = 10,
                     distribution: str = 'none',
                     seed: int = 0,
                     weight_reg_scale: Optional[float] = None) -> LossCtor:
  """Constructs a default loss suitable for classification or regression."""
  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: networks.EnnNoState) -> losses.LossFnNoState:
    # Construct L2 or Xent loss based on regression/classification.
    if prior.num_classes > 1:
      single_loss = losses.combine_single_index_losses_no_state_as_metric(
          train_loss=losses.XentLoss(prior.num_classes),
          extra_losses={'acc': losses.AccuracyErrorLoss(prior.num_classes)},
      )
    else:
      single_loss = losses.L2Loss()

    # Add bootstrapping
    boot_fn = data_noise.BootstrapNoise(enn, distribution, seed)
    single_loss = losses.add_data_noise_no_state(single_loss, boot_fn)

    loss_fn = losses.average_single_index_loss_no_state(single_loss,
                                                        num_index_samples)

    # Add L2 weight decay
    if weight_reg_scale:
      scale = (weight_reg_scale ** 2) / (2. * prior.num_train)
      loss_fn = losses.add_l2_weight_decay_no_state(loss_fn, scale=scale)
    return loss_fn
  return loss_ctor


def gaussian_regression_loss(num_index_samples: int,
                             noise_scale: float = 1,
                             l2_weight_decay: float = 0,
                             exclude_bias_l2: bool = True) -> LossCtor:
  """Add a matching Gaussian noise to the target y."""
  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: networks.EnnNoState) -> losses.LossFnNoState:
    """Add a matching Gaussian noise to the target y."""
    noise_std = noise_scale * prior.noise_std
    noise_fn = data_noise.GaussianTargetNoise(enn, noise_std)
    single_loss = losses.add_data_noise_no_state(losses.L2Loss(), noise_fn)
    loss_fn = losses.average_single_index_loss_no_state(single_loss,
                                                        num_index_samples)
    if l2_weight_decay != 0:
      if exclude_bias_l2:
        predicate = lambda module, name, value: name != 'b'
      else:
        predicate = lambda module, name, value: True
      loss_fn = losses.add_l2_weight_decay_no_state(loss_fn, l2_weight_decay,
                                                    predicate)
    return loss_fn
  return loss_ctor


def regularized_dropout_loss(num_index_samples: int = 10,
                             dropout_rate: float = 0.05,
                             scale: float = 1e-2,
                             tau: float = 1.0) -> LossCtor:
  """Constructs the special regularized loss of the paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (2015)."""
  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: networks.EnnNoState) -> losses.LossFnNoState:
    del enn  # Unused
    if prior.num_classes > 1:
      single_loss = losses.combine_single_index_losses_no_state_as_metric(
          train_loss=losses.XentLoss(prior.num_classes),
          extra_losses={'acc': losses.AccuracyErrorLoss(prior.num_classes)},
      )
    else:
      single_loss = losses.L2Loss()
    reg = (scale**2) * (1 - dropout_rate) / (2. * prior.num_train * tau)
    loss_fn = losses.average_single_index_loss_no_state(single_loss,
                                                        num_index_samples)
    return losses.add_l2_weight_decay_no_state(loss_fn, scale=reg)
  return loss_ctor


def combine_loss_prior_loss(loss_fn: losses.LossFnNoState,
                            prior_loss_fn: Optional[
                                losses.LossFnNoState] = None,
                            weight: float = 1.) -> losses.LossFnNoState:
  """Compatibility wrapper for deprecated prior_loss_fn interface."""
  if prior_loss_fn is None:
    return loss_fn
  else:
    return losses.combine_losses_no_state([
        losses.CombineLossConfigNoState(loss_fn, 'loss'),
        losses.CombineLossConfigNoState(prior_loss_fn, 'prior', weight),
    ])
