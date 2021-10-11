
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
"""A simple example running an ENN on the testbed."""

from absl import app
from absl import flags
from neural_testbed import leaderboard
from neural_testbed.agents import factories
from neural_testbed.experiments import experiment
from neural_testbed.leaderboard import sweep


# Option to configue the leaderboard problem instance.
# To do a *sweep* over all problem_ids pass --problem_id=SWEEP
flags.DEFINE_string('problem_id', 'classification_2d/0',
                    'ID for leaderboard GP.')

# Options for logging results to csv for evaluation later.
flags.DEFINE_string(
    'results_dir', '/tmp/neural_testbed', 'Where to store results as csv.')
flags.DEFINE_bool('overwrite_csv', True, 'Whether to overwrite existing csv.')

# Loading agent from factories, #sweep_id config within agent_name sweep
flags.DEFINE_string('agent_name', 'vanilla_ensemble', 'Agent to load')
flags.DEFINE_integer('sweep_id', -1, 'Agent within sweep, <0 gives default.')
flags.DEFINE_integer(
    'num_batches', -1,
    'override number of training batch. Mostly used for testing.')

FLAGS = flags.FLAGS


def run_single_problem(problem_id: str) -> str:
  """Evaluates the agent on a single problem instance."""
  # Load the problem via problem_id.
  problem = leaderboard.problem_from_id_csv(
      FLAGS.problem_id, FLAGS.results_dir, FLAGS.overwrite_csv)

  # Define the agent. Here we are constructing one of the benchmark agents
  # implemented in the factories package.
  paper_agent = factories.get_paper_agent(FLAGS.agent_name)
  if FLAGS.sweep_id < 0:
    config = paper_agent.default
    if FLAGS.num_batches > 0 and hasattr(config, 'num_batches'):
      config.num_batches = FLAGS.num_batches
  else:
    config = paper_agent.sweep()[FLAGS.sweep_id]
  agent = paper_agent.ctor(config)

  # Run the experiment and output the KL score.
  kl_quality = experiment.run(agent, problem)
  print(f'kl_quality={kl_quality}, write csv to {FLAGS.results_dir}')
  return problem_id


def main(_):
  if FLAGS.problem_id == 'SWEEP':
    # Perform a sweep over all the relevant problem_id for full evaluation.
    for problem_id in sweep.CLASSIFICATION_2D:
      run_single_problem(problem_id)

  else:
    # Run just a single problem_id.
    run_single_problem(FLAGS.problem_id)


if __name__ == '__main__':
  app.run(main)
