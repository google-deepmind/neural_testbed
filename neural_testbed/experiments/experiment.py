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
"""Common experiment loop for submitting to the testbed."""

from neural_testbed import base as testbed_base


def run(agent: testbed_base.TestbedAgent,
        problem: testbed_base.TestbedProblem) -> testbed_base.ENNQuality:
  """Run an agent on a given testbed problem."""
  enn_sampler = agent(problem.train_data, problem.prior_knowledge)
  return problem.evaluate_quality(enn_sampler)
