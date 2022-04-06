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
"""Tests whether leaderboard results match benchmark results."""

from absl.testing import absltest
from absl.testing import parameterized
from neural_testbed import base as testbed_base
from neural_testbed import leaderboard
from neural_testbed.agents.factories.sweeps import testbed_2d as factories
from neural_testbed.experiments import experiment

# TODO(author3): Store agent results in a csv. This allows more detailed tests.  # pylint: disable=line-too-long

_UNIFORM_AGENT_KLS = {
    'classification/0': 0.6931,
    'classification/100': 0.0967,
    'classification/200': 4.4106,
    'classification/300': 6.5033,
    'classification_2d/0': 0.6931,
    'classification_2d/100': 0.6105,
    'classification_2d/200': 0.0918,
    'classification_2d/300': 6.6814
}

_AVERAGE_AGENT_KLS = {
    'classification/0': 0.3133,
    'classification/100': 0.0584,
    'classification/200': 2.0021,
    'classification/300': 6.2137,
    'classification_2d/0': 0.3133,
    'classification_2d/100': 0.2667,
    'classification_2d/200': 0.0431,
    'classification_2d/300': 6.589
}


def get_agent_from_name(agent_name: str) -> testbed_base.EpistemicSampler:
  """Returns a testbed agent with default sweep from agent_name."""
  paper_agent = factories.get_paper_agent(agent_name)
  config = paper_agent.default
  agent = paper_agent.ctor(config)
  return agent


def run_agent_on_problem(
    agent: testbed_base.EpistemicSampler,
    problem_id: str,
) -> float:
  """Evaluates the agent on a single problem instance and returns the kl estimate."""
  # Load the problem via problem_id.
  problem = leaderboard.problem_from_id_csv(problem_id)

  # Run the experiment and output the KL score.
  kl_quality = experiment.run(agent, problem)
  return kl_quality.kl_estimate


class RunTest(parameterized.TestCase):

  @parameterized.product(problem_id=list(_UNIFORM_AGENT_KLS.keys()))
  def test_uniform_agent(self, problem_id: str):
    """Tests uniform probabilities agent on a set of testbed problems.

    Since uniform agent does not use train data, this test checks
      1) Generative process for test data
      2) KL calculation

    Args:
      problem_id: a testbed problem.
    """
    agent = get_agent_from_name('baseline:uniform_class_probs')
    kl_estimate = run_agent_on_problem(agent, problem_id)
    expected_kl_estimate = _UNIFORM_AGENT_KLS[problem_id]
    self.assertAlmostEqual(
        kl_estimate, expected_kl_estimate,
        msg=(f'Expected KL estimate to be {expected_kl_estimate} ',
             f'but received {kl_estimate}'),
        delta=0.01)

  @parameterized.product(problem_id=list(_AVERAGE_AGENT_KLS.keys()))
  def test_average_agent(self, problem_id: str):
    """Tests average probabilities agent on a set of testbed problems.

    Since average agent uses train data, this test checks
      1) Generative process for train data
      1) Generative process for test data
      2) KL calculation

    Args:
      problem_id: a testbed problem.
    """
    agent = get_agent_from_name('baseline:average_class_probs')
    kl_estimate = run_agent_on_problem(agent, problem_id)
    expected_kl_estimate = _AVERAGE_AGENT_KLS[problem_id]
    self.assertAlmostEqual(
        kl_estimate, expected_kl_estimate,
        msg=(f'Expected KL estimate to be {expected_kl_estimate} ',
             f'but received {kl_estimate}'),
        delta=0.01)


if __name__ == '__main__':
  absltest.main()
