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

"""A minimalist wrapper around ENN experiment for testbed submission."""

import dataclasses
from typing import Callable, Dict, Optional, Union

from acme.utils import loggers
import chex
from enn import datasets
from enn import networks
from enn import supervised
from enn import utils
import jax
from neural_testbed import base as testbed_base
from neural_testbed import likelihood
from neural_testbed.agents import enn_losses
import optax


# Allow for either an integer num_batches or determined from prior
_BatchStrategy = Union[int, Callable[[testbed_base.PriorKnowledge], int]]


@dataclasses.dataclass
class VanillaEnnConfig:
  """Configuration options for the VanillaEnnAgent."""
  enn_ctor: enn_losses.EnnCtor
  loss_ctor: enn_losses.LossCtor = enn_losses.default_enn_loss()
  optimizer: optax.GradientTransformation = optax.adam(1e-3)
  num_batches: _BatchStrategy = 1000
  # TODO(author2): Complete prior loss refactor --> MultilossExperiment
  prior_loss_ctor: Optional[enn_losses.LossCtor] = None
  prior_loss_freq: int = 10
  seed: int = 0
  batch_size: int = 100
  center_train_data: bool = False
  eval_batch_size: Optional[int] = None
  logger: Optional[loggers.Logger] = None
  train_log_freq: Optional[int] = None
  eval_log_freq: Optional[int] = None


def extract_enn_sampler(
    experiment: supervised.BaseExperiment) -> testbed_base.EpistemicSampler:
  def enn_sampler(x: chex.Array, key: chex.PRNGKey) -> chex.Array:
    """Generate a random sample from posterior distribution at x."""
    net_out = experiment.predict(x, key)
    return networks.parse_net_output(net_out)
  return jax.jit(enn_sampler)


@dataclasses.dataclass
class VanillaEnnAgent(testbed_base.TestbedAgent):
  """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
  config: VanillaEnnConfig
  eval_datasets: Optional[Dict[str, datasets.ArrayBatchIterator]] = None
  experiment: Optional[supervised.BaseExperiment] = None

  def __call__(
      self,
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge,
  ) -> testbed_base.EpistemicSampler:
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
    enn = self.config.enn_ctor(prior)
    if self.config.center_train_data:
      enn = networks.make_centered_enn(enn, data.x)

    enn_data = datasets.ArrayBatch(x=data.x, y=data.y)
    dataset = utils.make_batch_iterator(
        enn_data, self.config.batch_size, self.config.seed)

    # TODO(author2): Complete prior loss refactor --> MultilossExperiment
    trainers = []
    trainers.append(supervised.MultilossTrainer(
        loss_fn=self.config.loss_ctor(prior, enn),
        dataset=dataset,
    ))
    if self.config.prior_loss_ctor is not None:
      trainers.append(supervised.MultilossTrainer(
          loss_fn=self.config.prior_loss_ctor(prior, enn),
          dataset=dataset,
          should_train=lambda step: step % self.config.prior_loss_freq == 0,
          name='prior_loss',
      ))

    # Parse number of training batches from config _BatchStrategy
    if isinstance(self.config.num_batches, int):
      num_batches = self.config.num_batches
    else:
      num_batches = self.config.num_batches(prior)

    self.experiment = supervised.MultilossExperiment(
        enn=enn,
        trainers=trainers,
        optimizer=self.config.optimizer,
        seed=self.config.seed,
        logger=self.config.logger,
        train_log_freq=logging_freq(
            num_batches, log_freq=self.config.train_log_freq),
        eval_datasets=self.eval_datasets,
        eval_log_freq=logging_freq(
            num_batches, log_freq=self.config.eval_log_freq),
    )

    # Train agent and return the ENN
    self.experiment.train(num_batches)
    return extract_enn_sampler(self.experiment)


def make_learning_curve_enn_agent(
    config: VanillaEnnConfig,
    problem: testbed_base.TestbedProblem,
    num_test: int = 1000,
    seed: int = 0
) -> VanillaEnnAgent:
  """Constructs an agent with privileged access to testing data.

  This constructor will look inside the testbed problem and try to extract the
  testing data, for periodic evaluation within the *experiment* dataframe of
  the agent. This should allow us to produce learning curves on train/test.
  However, it is slightly *hacky*... so use at your own risk!

  Args:
    config: options for the vanilla ENN agent.
    problem: problem instance, ideally it should contain a SampleBasedTestbed.
    num_test: number of testing datapoints for the "test" dataset.
    seed: an integer seed.

  Returns:
    VanillaEnnAgent with internal logging of train/test.
  """
  problem = getattr(problem, 'problem', problem)
  if isinstance(problem, likelihood.SampleBasedTestbed):
    # Convert the data to enn batch format
    train_data = datasets.ArrayBatch(
        x=problem.train_data.x, y=problem.train_data.y
    )

    # Generate a sample-based test dataset with num_test samples.
    def gen_test(key: chex.PRNGKey) -> testbed_base.Data:
      data, _ = problem.data_sampler.test_data(key)
      return testbed_base.Data(x=data.x[0, :], y=data.y[0, :])

    test_keys = jax.random.split(jax.random.PRNGKey(seed), num_test)
    test_data = jax.lax.map(gen_test, test_keys)
    test_data = datasets.ArrayBatch(x=test_data.x, y=test_data.y)

    # Pass out eval_datasets to experiment.
    eval_datasets = {
        'train': utils.make_batch_iterator(train_data, config.eval_batch_size),
        'test': utils.make_batch_iterator(test_data, config.eval_batch_size),
    }
  else:
    print(f'WARNING: problem={problem} is not SampleBasedTestbed.')
    eval_datasets = None
  return VanillaEnnAgent(config, eval_datasets)


def _round_to_integer(x: float) -> int:
  """Utility function to round a float to integer, or 1 if it would be 0."""
  x = int(x)
  if x == 0:
    return 1
  else:
    return x


def logging_freq(num_batches: int,
                 num_points: int = 30,
                 log_freq: Optional[int] = None) -> int:
  """Computes a logging frequency from num_batches, optionally log_freq."""
  if log_freq is None:
    log_freq = _round_to_integer(num_batches / num_points)
  return log_freq
