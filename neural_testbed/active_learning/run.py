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
"""Example running an ENN on active learning task."""

from absl import app
from absl import flags
from acme.utils import loggers
from enn import active_learning
from neural_testbed.active_learning import experiment
from neural_testbed.agents.factories.sweeps import testbed_2d as factories
from neural_testbed.bandit import agents


# ENN training
_AGENT_ID = flags.DEFINE_string('agent_id', 'mlp',
                                'Which benchmark agent to run.')

# Action priority
_PRIORITY = flags.DEFINE_string('priority', 'entropy', 'How to prioritize data')

# Active learning
_INPUT_DIM = flags.DEFINE_integer('input_dim', 10, 'Input dimension')
_TEMPERATURE = flags.DEFINE_float('temperature', 0.1, 'Temperature')
_NUM_ACTIONS = flags.DEFINE_integer('num_actions', 20, 'Number of actions')
_NUM_STEPS = flags.DEFINE_integer('num_steps', 10_000, 'Number of timesteps')
_SEED = flags.DEFINE_integer('seed', 0, 'Bandit seed')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'batch size in training')
_STEPS_PER_OBS = flags.DEFINE_integer('steps_per_obs', 1,
                                      'sgds per observation')


def main(_):
  # Override this config for different ENNs... must be a VanillaEnnAgent
  paper_agent = factories.get_paper_agent(_AGENT_ID.value)

  # Convert testbed agent to an active learning agent
  config, l2_weight_decay = agents.make_config_l2_for_bandit(
      paper_agent=paper_agent,
      temperature=_TEMPERATURE.value,
      seed=_SEED.value,
  )

  # Run the active learning experiment with appropriate logging
  trainer = experiment.ActiveLearning(
      enn_config=config,
      priority_fn_ctor=active_learning.get_priority_fn_ctor(_PRIORITY.value),
      input_dim=_INPUT_DIM.value,
      num_actions=_NUM_ACTIONS.value * _INPUT_DIM.value,
      temperature=_TEMPERATURE.value,
      seed=_SEED.value,
      steps_per_obs=_STEPS_PER_OBS.value,
      logger=loggers.make_default_logger('results', time_delta=0),
      should_log=lambda x: x % 10 == 0,
      batch_size=_BATCH_SIZE.value,
      l2_weight_decay=l2_weight_decay,
  )
  trainer.run(_NUM_STEPS.value)

if __name__ == '__main__':
  app.run(main)
