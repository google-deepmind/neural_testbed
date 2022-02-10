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
"""Example running an ENN on Thompson bandit task."""

from absl import app
from absl import flags
from neural_testbed.agents.factories.sweeps import testbed_2d as factories
from neural_testbed.bandit import agents
from neural_testbed.bandit import thompson

# ENN training
flags.DEFINE_string('agent_id', 'ensemble+', 'Which benchmark agent to run.')

# Bandit problem
flags.DEFINE_integer('input_dim', 2, 'Input dimension')
flags.DEFINE_float('temperature', 0.1, 'Temperature')
flags.DEFINE_integer('num_actions', 50, 'Number of actions')
flags.DEFINE_integer('num_steps', 10_000, 'Number of timesteps')
flags.DEFINE_integer('seed', 0, 'Bandit seed')
flags.DEFINE_integer('steps_per_obs', 1, 'sgds per observation')

FLAGS = flags.FLAGS


def main(_):
  # Override this config for different ENNs... must be a VanillaEnnAgent
  paper_agent = factories.get_paper_agent(FLAGS.agent_id)

  # Convert testbed agent to sequential decision agent
  config, l2_weight_decay = agents.make_config_l2_for_bandit(
      paper_agent=paper_agent,
      temperature=FLAGS.temperature,
      seed=FLAGS.seed,
  )

  # Run the bandit experiment with appropriate logging
  experiment = thompson.ThompsonEnnBandit(
      enn_config=config,
      input_dim=FLAGS.input_dim,
      num_actions=FLAGS.num_actions * FLAGS.input_dim,
      temperature=FLAGS.temperature,
      seed=FLAGS.seed,
      steps_per_obs=FLAGS.steps_per_obs,
      l2_weight_decay=l2_weight_decay,
  )
  log_freq = int(FLAGS.num_steps / 100)
  if log_freq == 0:
    log_freq = 1
  experiment.run(FLAGS.num_steps, log_freq)

if __name__ == '__main__':
  app.run(main)

