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
"""Factory methods for sgmcmc agent."""

import dataclasses

from absl import logging
import chex
from enn import base as enn_base
from enn import losses
from enn import networks
from enn import supervised
from enn import utils as enn_utils
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.agents.factories import preconditioner as pre
from neural_testbed.agents.factories import sgld_optimizer
from neural_testbed.agents.factories import utils


@dataclasses.dataclass
class SGMCMCConfig:
  """Config Class for SGMCMC."""
  learning_rate: float = 0.0001  # Learning rate for optimizers
  prior_variance: float = 0.1  # Variance of Gaussian prior
  alg_temperature: float = 1  # Temperature parameter for SGLD
  momentum_decay: float = 0.9  # Momentum decay parameter for SGLD
  preconditioner: str = 'None'  # Choice of preconditioner; None or RMSprop
  num_hidden: int = 50  # Hidden units in network
  num_batches: int = 500000  # Number of SGD steps
  burn_in_time: int = 100000  # Burn in time for MCMC sampling
  model_saving_frequency: int = 1000  # Frequency of saving models
  adaptive_prior_variance: bool = False  # Scale prior_variance with dimension
  seed: int = 0  # Initialization seed


# Choice of using preconditioner
def get_preconditioner(config: SGMCMCConfig):
  if config.preconditioner == 'None':
    preconditioner = None
  else:
    preconditioner = pre.get_rmsprop_preconditioner()
  return preconditioner


# ENN sampler for MCMC
def extract_enn_sampler(enn: enn_base.EpistemicNetwork,
                        params_list) -> testbed_base.EpistemicSampler:
  """ENN sampler for MCMC."""
  def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
    """Generate a random sample from posterior distribution at x."""
    # pylint: disable=cell-var-from-loop
    param_index = jax.random.randint(key, [], 0, len(params_list))
    fns = [lambda x, w=p: enn.apply(w, x, 0) for p in params_list]
    out = jax.lax.switch(param_index, fns, x)
    return enn_utils.parse_net_output(out)
  return jax.jit(enn_sampler)


def make_agent(config: SGMCMCConfig):
  """Factory method to create a sgmcmc agent."""

  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    return networks.make_einsum_ensemble_mlp_enn(
        output_sizes=[config.num_hidden, config.num_hidden, prior.num_classes],
        num_ensemble=1,
        nonzero_bias=False,
    )

  def make_loss(prior: testbed_base.PriorKnowledge) -> enn_base.LossFn:
    single_loss = losses.combine_single_index_losses_as_metric(
        # This is the loss you are training on.
        train_loss=losses.XentLoss(prior.num_classes),
        # We will also log the accuracy in classification.
        extra_losses={'acc': losses.AccuracyErrorLoss(prior.num_classes)},
    )
    loss_fn = losses.average_single_index_loss(single_loss, 1)
    # Gaussian prior can be interpreted as a L2-weight decay.
    prior_variance = config.prior_variance
    # Scale prior_variance for large input_dim
    if config.adaptive_prior_variance and prior.input_dim >= 100:
      prior_variance *= 2
    scale = (1 / prior_variance) * jnp.sqrt(
        prior.temperature) * prior.input_dim / prior.num_train
    loss_fn = losses.add_l2_weight_decay(loss_fn, scale=scale)
    return loss_fn

  log_freq = int(config.num_batches / 50) or 1

  def sgd_agent(
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge,
  ) -> testbed_base.EpistemicSampler:
    """Train a MLP via SGMCMC."""
    preconditioner = get_preconditioner(config)
    optimizer_sgld = sgld_optimizer.sgld_gradient_update(
        config.learning_rate,
        momentum_decay=config.momentum_decay,
        seed=0,
        preconditioner=preconditioner,
        temperature=config.alg_temperature/prior.num_train)

    # Define the experiment
    sgd_experiment = supervised.Experiment(
        enn=make_enn(prior),
        loss_fn=make_loss(prior),
        optimizer=optimizer_sgld,
        dataset=utils.make_iterator(data, batch_size=100),
        train_log_freq=log_freq,
    )

    # Train the agent
    params_list = []
    step = 0
    for _ in range(config.num_batches):
      step += 1
      sgd_experiment.train(1)
      # Save the model every model_saving_frequency steps
      if step >= config.burn_in_time and ((step - config.burn_in_time) %
                                          config.model_saving_frequency == 0):
        params_list.append(sgd_experiment.state.params)
        logging.info('Saving params at step %d.', step)

    return extract_enn_sampler(make_enn(prior), params_list)
  return sgd_agent
