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
"""Active learning evaluation of ENN agent on testbed problem."""

import functools
import typing as tp

from acme.utils import loggers
import chex
from enn import active_learning
from enn import base as enn_base
from enn import losses
from enn import networks
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed import agents
from neural_testbed import base as testbed_base
from neural_testbed import generative
from neural_testbed import likelihood
from neural_testbed.bandit import replay
from neural_testbed.leaderboard import sweep
import optax


class ActiveLearning:
  """Active learning experiment."""

  def __init__(
      self,
      enn_config: agents.VanillaEnnConfig,
      priority_fn_ctor: active_learning.PriorityFnCtor,
      input_dim: int,
      num_actions: int,
      logit_ctor: tp.Optional[sweep.LogitCtor] = None,
      temperature: float = 1,
      steps_per_obs: int = 1,
      logger: tp.Optional[loggers.Logger] = None,
      should_log: tp.Callable[[int], bool] = lambda x: True,
      batch_size: int = 16,
      l2_weight_decay: float = 1,
      replay_capacity: int = 10_000,
      learning_rate: float = 1e-3,
      seed: int = 0,
  ):
    """Initializes an active learning experiment."""

    # Initializing the agent internals
    prior = testbed_base.PriorKnowledge(
        input_dim=input_dim,
        num_train=100,
        num_classes=2,
        tau=1,
        layers=2,
        temperature=temperature,
    )
    self.observed_actions = set()
    self.enn = enn_config.enn_ctor(prior)
    batch_fwd = networks.make_batch_fwd(self.enn)
    self.priority_fn = jax.jit(priority_fn_ctor(batch_fwd))
    loss_fn = enn_config.loss_ctor(prior, self.enn)
    loss_fn = functools.partial(loss_fn, self.enn)

    def predicate(module_name: str, name: str, value) -> bool:
      del name, value
      return 'prior' not in module_name

    def loss_with_decay(
        params: hk.Params,
        state: hk.State,
        batch: enn_base.Batch,
        key: chex.PRNGKey) -> enn_base.LossOutput:
      # Adding annealing l2 weight decay manually
      data_loss, (state, metrics) = loss_fn(params, state, batch, key)
      l2_weight = losses.l2_weights_with_predicate(params, predicate)
      metrics['l2_weight'] = l2_weight
      decay_loss = l2_weight_decay * l2_weight / batch.extra['num_steps']
      return data_loss + decay_loss, (state, metrics)
    self._loss_with_decay = jax.jit(loss_with_decay)

    optimizer = optax.adam(learning_rate)

    # Perform an SGD step on a batch of data
    def sgd_step(
        params: hk.Params,
        opt_state: optax.OptState,
        batch: enn_base.Batch,
        key: chex.PRNGKey,
    ) -> tp.Tuple[hk.Params, optax.OptState]:
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

    # Creating the testing environment
    self.test_problem = _make_test_problem(
        logit_fn=logit_fn, prior=prior, input_dim=input_dim, key=next(self.rng))

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
    self.should_log = should_log
    self.num_steps = 0

    def select_action(params: hk.Params,
                      key: chex.PRNGKey) -> tp.Dict[str, chex.Array]:
      priority_key, noise_key, selection_key = jax.random.split(key, 3)
      # Randomly generate rewards for each action.
      rewards = jax.random.bernoulli(noise_key, self.probs)
      # Get the priority score for each action.
      batch = enn_base.Batch(x=self.actions, y=rewards)
      dummy_state = {}
      priorities, _ = self.priority_fn(params, dummy_state, batch, priority_key)
      # Pick an action with the highest priority (with additional random noise).
      action = _random_argmax(priorities, selection_key)
      # Calculate the regret.
      chosen_prob = self.probs[action]
      reward = rewards[action]
      regret = self.max_prob - chosen_prob
      return {
          'action': action,
          'reward': reward,
          'regret': regret,
          'chosen_prob': chosen_prob,  # for debugging
      }
    self._select_action = jax.jit(select_action)

  def run(self, num_steps: int):
    """Runs a TS experiment for num_steps."""
    for _ in range(num_steps):
      self.num_steps += 1
      action = self.step()
      self.observed_actions.add(action)

      if self.should_log(self.num_steps):
        # Evaluate the ENN on test data
        def enn_sampler(x: chex.Array, key: chex.PRNGKey) -> chex.Array:
          index = self.enn.indexer(key)
          net_out, unused_state = self.enn.apply(self.params, {}, x, index)
          return networks.parse_net_output(net_out)
        enn_quality = self.test_problem.evaluate_quality(jax.jit(enn_sampler))
        results = {
            'kl_estimate': float(enn_quality.kl_estimate),
            'num_steps': self.num_steps,
            'num_data': len(self.observed_actions),
        }
        results.update(_clean_results(enn_quality.extra))
        self.logger.write(results)

      for _ in range(self._steps_per_obs):
        if self.num_steps >= 1:
          self.params, self.opt_state = self._sgd_step(
              self.params, self.opt_state, self._get_batch(), next(self.rng))

  def step(self) -> int:
    """Selects action, update replay and return the selected action."""
    results = self._select_action(self.params, next(self.rng))
    self.replay.add([
        self.actions[results['action']],
        jnp.ones([1]) * results['reward'],
        jnp.ones([1], dtype=jnp.int64) * self.num_steps,
    ])
    return int(results['action'])

  def _get_batch(self) -> enn_base.Batch:
    """Samples a batch from the replay."""
    actions, rewards, indices = self.replay.sample(self._batch_size)
    return enn_base.Batch(
        actions, rewards, indices, extra={'num_steps': self.num_steps})


def _make_test_problem(logit_fn: generative.LogitFn,
                       prior: testbed_base.PriorKnowledge,
                       input_dim: int,
                       key: chex.PRNGKey,
                       num_classes: int = 2) -> likelihood.SampleBasedTestbed:
  """Makes the test environment."""
  sampler_key, kl_key = jax.random.split(key)
  # Defining dummy values for x_train_generator and num_train. These values are
  # not used as we only use data_sampler to make test data.
  dummy_x_train_generator = generative.make_gaussian_sampler(input_dim)
  dummy_num_train = 10
  data_sampler = generative.ClassificationEnvLikelihood(
      logit_fn=logit_fn,
      x_train_generator=dummy_x_train_generator,  # UNUSED
      x_test_generator=generative.make_gaussian_sampler(input_dim),
      num_train=dummy_num_train,  # UNUSED
      key=sampler_key,
      tau=1,
  )
  sample_based_kl = likelihood.CategoricalKLSampledXSampledY(
      num_test_seeds=1000,
      num_enn_samples=1000,
      key=kl_key,
      num_classes=num_classes,
  )
  sample_based_kl = likelihood.add_classification_accuracy_ece(
      sample_based_kl,
      num_test_seeds=1000,
      num_enn_samples=100,
      num_classes=num_classes,
  )
  return likelihood.SampleBasedTestbed(
      data_sampler, sample_based_kl, prior)


def _random_argmax(vals: chex.Array,
                   key: chex.PRNGKey,
                   scale: float = 1e-5) -> int:
  """Selects argmax with additional random noise."""
  noise = jax.random.normal(key, vals.shape)
  return jnp.argmax(vals + scale * noise, axis=0)


def _clean_results(results: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
  """Cleans the results for logging (can't log jax arrays)."""
  def clean_result(value: tp.Any) -> tp.Any:
    value = loggers.to_numpy(value)
    if isinstance(value, chex.ArrayNumpy) and value.size == 1:
      value = float(value)
    return value

  for key, value in results.items():
    results[key] = clean_result(value)

  return results
