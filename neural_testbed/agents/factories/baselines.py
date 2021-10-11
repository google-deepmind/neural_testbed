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
"""Factory methods for baseline agents."""

import dataclasses
from typing import Sequence

import chex
from enn import base as enn_base
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.agents import enn_agent
from neural_testbed.agents.factories import base as factories_base
from neural_testbed.agents.factories import ensemble


@dataclasses.dataclass
class MLPConfig:
  adaptive_weight_scale: bool = True  # Whether to scale with prior
  l2_weight_decay: float = 1.  # Weight decay
  hidden_sizes: Sequence[int] = (50, 50)  # Hidden sizes for the neural network
  num_batches: int = 1000  # Number of SGD steps
  seed: int = 0  # Initialization seed


def make_mlp_agent(config: MLPConfig) -> enn_agent.VanillaEnnAgent:
  """Factory method to create a baseline MLP agent."""
  config = ensemble.VanillaEnsembleConfig(
      num_ensemble=1,
      l2_weight_decay=config.l2_weight_decay,
      adaptive_weight_scale=config.adaptive_weight_scale,
      hidden_sizes=config.hidden_sizes,
      num_batches=config.num_batches,
      seed=config.seed)
  return ensemble.make_agent(config)


def mlp_sweep() -> Sequence[MLPConfig]:
  sweep = []
  for adaptive_weight_scale in [True, False]:
    for l2_weight_decay in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
      sweep.append(MLPConfig(
          l2_weight_decay=l2_weight_decay,
          adaptive_weight_scale=adaptive_weight_scale,
      ))
  return tuple(sweep)


def mlp_paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=MLPConfig(),
      ctor=make_mlp_agent,
      sweep=mlp_sweep)


@dataclasses.dataclass
class LogisticRegressionConfig:
  adaptive_weight_scale: bool = True  # Whether to scale with prior
  l2_weight_decay: float = 1.  # Weight decay
  num_batches: int = 1000  # Number of SGD steps
  seed: int = 0  # Initialization seed


def make_logistic_regression_agent(
    config: LogisticRegressionConfig) -> enn_agent.VanillaEnnAgent:
  """Factory method to create a baseline logistic regression agent."""
  config = ensemble.VanillaEnsembleConfig(
      num_ensemble=1,
      l2_weight_decay=config.l2_weight_decay,
      adaptive_weight_scale=config.adaptive_weight_scale,
      hidden_sizes=(),
      num_batches=config.num_batches,
      seed=config.seed)
  return ensemble.make_agent(config)


def logistic_regression_sweep() -> Sequence[LogisticRegressionConfig]:
  sweep = []
  for adaptive_weight_scale in [True, False]:
    for l2_weight_decay in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
      sweep.append(LogisticRegressionConfig(
          l2_weight_decay=l2_weight_decay,
          adaptive_weight_scale=adaptive_weight_scale,
      ))
  return tuple(sweep)


def logistic_regression_paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=LogisticRegressionConfig(),
      ctor=make_logistic_regression_agent,
      sweep=logistic_regression_sweep)


@dataclasses.dataclass
class DummyConfig:
  seed: int = 0  # Initialization seed


def dummy_sweep() -> Sequence[DummyConfig]:
  return tuple([DummyConfig()])


def make_uniform_class_probs_agent(
    config: DummyConfig) -> testbed_base.TestbedAgent:
  """Factory method to create a baseline uniform class probability agent."""
  del config
  def make_agent(
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge,
  ) -> testbed_base.EpistemicSampler:
    """Ignores the input and always outputs equal logits for all classes."""
    del data  # data does not affect the baseline agent.
    def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
      del key  # key does not affect the baseline agent.
      return jnp.ones([x.shape[0], prior.num_classes]) / prior.num_classes
    return enn_sampler
  return make_agent


def uniform_class_probs_paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=DummyConfig(),
      ctor=make_uniform_class_probs_agent,
      sweep=dummy_sweep)


def make_average_class_probs_agent(
    config: DummyConfig) -> testbed_base.TestbedAgent:
  """Factory method to create a baseline average class probability agent."""
  del config
  def make_agent(
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge) -> testbed_base.EpistemicSampler:
    """Calculates the frequency of each class and outputs the class frequency."""
    counts = jnp.array([
        jnp.count_nonzero(data.y == label) for label in range(prior.num_classes)
    ])
    average_probs = counts / prior.num_train
    def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
      del key  # key does not affect the baseline agent.
      return jnp.repeat(average_probs[None, :], x.shape[0], axis=0)
    return enn_sampler
  return make_agent


def average_class_probs_paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=DummyConfig(),
      ctor=make_average_class_probs_agent,
      sweep=dummy_sweep)


def make_prior_agent(
    config: DummyConfig) -> testbed_base.TestbedAgent:
  """Factory method to create an agent that uses prior knowledge but ignores data."""
  del config
  def make_agent(
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge) -> testbed_base.EpistemicSampler:
    """Samples an MLP according to the generative model."""
    del data
    hidden = 50
    def net_fn(x: chex.Array) -> chex.Array:
      """Defining the generative model MLP."""
      y = hk.Linear(
          output_size=hidden,
          b_init=hk.initializers.RandomNormal(1./jnp.sqrt(prior.input_dim)),
      )(x)
      y = jax.nn.relu(y)
      y = hk.Linear(hidden)(y)
      y = jax.nn.relu(y)
      return hk.Linear(prior.num_classes)(y)

    transformed = hk.without_apply_rng(hk.transform(net_fn))

    def sampler(x: chex.Array, key: chex.PRNGKey) -> chex.Array:
      params = transformed.init(key, x)
      return transformed.apply(params, x) / prior.temperature

    return jax.jit(sampler)
  return make_agent


def prior_paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=DummyConfig(),
      ctor=make_prior_agent,
      sweep=dummy_sweep)
