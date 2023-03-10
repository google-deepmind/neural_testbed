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
"""Thompson sampling evaluation of ENN agent on bandit task."""

import functools
from typing import Dict, Optional, Tuple

from acme.utils import loggers
import chex
from enn import base as enn_base
from enn import datasets
from enn import losses
from enn import networks
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed import agents
from neural_testbed import base as testbed_base
from neural_testbed import generative
from neural_testbed.bandit import replay
from neural_testbed.leaderboard import sweep
import optax


class ThompsonEnnBandit:
  """Experiment of Thompson sampling bandit."""

  def __init__(
      self,
      enn_config: agents.VanillaEnnConfig,
      input_dim: int,
      num_actions: int,
      logit_ctor: Optional[sweep.LogitCtor] = None,
      temperature: float = 1,
      steps_per_obs: int = 1,
      logger: Optional[loggers.Logger] = None,
      batch_size: int = 128,
      l2_weight_decay: float = 1,
      replay_capacity: int = 10_000,
      learning_rate: float = 1e-3,
      seed: int = 0,
  ):
    """Initialize a Thompson Sampling experiment."""

    # Initializing the agent internals
    prior = testbed_base.PriorKnowledge(
        input_dim=input_dim,
        num_train=100,
        num_classes=2,
        tau=1,
        layers=2,
        temperature=temperature,
    )
    self.enn = enn_config.enn_ctor(prior)
    loss_fn = enn_config.loss_ctor(prior, self.enn)
    loss_fn = functools.partial(loss_fn, self.enn)

    def predicate(module_name: str, name: str, value) -> bool:
      del name, value
      return 'prior' not in module_name

    def loss_with_decay(
        params: hk.Params,
        state: hk.State,
        batch: datasets.ArrayBatch,
        key: chex.PRNGKey) -> enn_base.LossOutput:
      # Adding annealing l2 weight decay manually
      data_loss, (state, metrics) = loss_fn(params, state, batch, key)
      l2_weight = losses.l2_weights_with_predicate(params, predicate)
      metrics['l2_weight'] = l2_weight
      decay_loss = l2_weight_decay * l2_weight / batch.extra['num_steps']
      return data_loss + decay_loss, (state, metrics)
    self._loss_with_decay = jax.jit(loss_with_decay)

    optimizer = optax.adam(learning_rate)

    # Forward network at random index
    def forward(params: hk.Params,
                inputs: chex.Array,
                key: chex.PRNGKey) -> chex.Array:
      index = self.enn.indexer(key)
      unused_state = {}
      out, unused_state = self.enn.apply(params, unused_state, inputs, index)
      return out
    self._forward = jax.jit(forward)

    # Perform an SGD step on a batch of data
    def sgd_step(
        params: hk.Params,
        opt_state: optax.OptState,
        batch: datasets.ArrayBatch,
        key: chex.PRNGKey,
    ) -> Tuple[hk.Params, optax.OptState]:
      unused_state = {}
      grads, _ = jax.grad(
          loss_with_decay, has_aux=True)(params, unused_state, batch, key)
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state
    self._sgd_step = jax.jit(sgd_step)

    # Generating the underlying function
    self.rng = hk.PRNGSequence(seed)
    self.actions = jax.random.normal(next(self.rng), [num_actions, input_dim])

    # Create the logit_fn
    if logit_ctor is None:
      logit_fn = generative.make_2layer_mlp_logit_fn(
          input_dim=input_dim,
          temperature=temperature,
          hidden=50,
          num_classes=2,
          key=next(self.rng),
      )
    else:
      logit_fn = logit_ctor(next(self.rng))
    logits = logit_fn(self.actions)

    # Vector of probabilities of rewards for each action
    self.probs = jax.nn.softmax(logits)[:, 1]
    chex.assert_shape(self.probs, [num_actions])
    self.max_prob = jnp.max(self.probs)

    # Initializing the network
    index = self.enn.indexer(next(self.rng))
    self.params, self.network_state = self.enn.init(
        next(self.rng), self.actions, index)
    self.opt_state = optimizer.init(self.params)
    self._steps_per_obs = steps_per_obs
    self._temperature = temperature
    self._batch_size = batch_size
    self.l2_weight_decay = l2_weight_decay
    self.replay = replay.Replay(capacity=replay_capacity)
    self.logger = (
        logger or loggers.make_default_logger('experiment', time_delta=0))
    self.num_steps = 0
    self.total_regret = 0

    def select_action(params: hk.Params,
                      key: chex.PRNGKey) -> Dict[str, chex.Array]:
      net_key, noise_key, selection_key = jax.random.split(key, 3)
      net_out = forward(params, self.actions, net_key)
      logits = networks.parse_net_output(net_out)
      probs = jax.nn.softmax(logits)[:, 1]
      action = _random_argmax(probs, selection_key)
      chosen_prob = self.probs[action]
      reward = jax.random.bernoulli(noise_key, chosen_prob)
      regret = self.max_prob - chosen_prob
      return {
          'action': action,
          'reward': reward,
          'regret': regret,
          'chosen_prob': chosen_prob,  # for debugging
      }
    self._select_action = jax.jit(select_action)

  def run(self, num_steps: int, log_freq: int = 1):
    """Run a TS experiment for num_steps."""
    for _ in range(num_steps):
      self.num_steps += 1
      regret = self.step()
      self.total_regret += regret
      if self.num_steps % log_freq == 0:
        self.logger.write({
            'total_regret': self.total_regret,
            't': self.num_steps,
            'ave_regret': self.total_regret / self.num_steps,
            'regret': regret,
        })
      for _ in range(self._steps_per_obs):
        if self.num_steps >= 1:
          self.params, self.opt_state = self._sgd_step(
              self.params, self.opt_state, self._get_batch(), next(self.rng))

  def step(self) -> float:
    """Select action, update replay and return the regret."""
    results = self._select_action(self.params, next(self.rng))
    self.replay.add([
        self.actions[results['action']],
        jnp.ones([1]) * results['reward'],
        jnp.ones([1], dtype=jnp.int64) * self.num_steps,
    ])
    return float(results['regret'])

  def _get_batch(self) -> datasets.ArrayBatch:
    actions, rewards, indices = self.replay.sample(self._batch_size)
    return datasets.ArrayBatch(
        x=actions,
        y=rewards,
        data_index=indices,
        extra={'num_steps': self.num_steps},
    )


def _random_argmax(
    vals: chex.Array, key: chex.PRNGKey, scale: float = 1e-7
) -> int:
  """Select argmax with additional random noise."""
  noise = jax.random.uniform(key, vals.shape)
  return jnp.argmax(vals + scale * noise, axis=0)
