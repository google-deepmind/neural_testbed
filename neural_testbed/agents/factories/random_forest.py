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
"""Factory methods for Random Forest baseline agent."""

import dataclasses

from enn import base as enn_base
from neural_testbed import base as testbed_base
import numpy as np
from sklearn import ensemble


@dataclasses.dataclass
class RandomForestConfig:
  n_estimators: int = 100  # Number of elements in random forest
  criterion: str = 'gini'  # Splitting criterion 'gini' or 'entropy'


def make_agent(config: RandomForestConfig) -> testbed_base.TestbedAgent:
  """Factory method to create a random forest agent."""

  def random_forest_agent(
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge,
  ) -> testbed_base.EpistemicSampler:
    # sklearn cannot handle instances with no samples of that class
    # so we add a fake sample for every class here as a hack
    new_x = np.zeros((prior.num_classes, data.x.shape[1]))
    new_y = np.expand_dims(np.arange(prior.num_classes), axis=1)
    data = testbed_base.Data(
        np.concatenate((data.x, new_x), axis=0),
        np.concatenate((data.y, new_y), axis=0))

    random_forest = ensemble.RandomForestClassifier(
        n_estimators=config.n_estimators, criterion=config.criterion)
    random_forest.fit(data.x, np.ravel(data.y))

    # Ensure that the number of classes is correct
    # (this will fail if the fake data isn't added above)
    assert len(random_forest.classes_) == prior.num_classes

    def enn_sampler(x: enn_base.Array, seed: int = 0) -> enn_base.Array:
      del seed  # seed does not affect the random_forest agent.
      probs = random_forest.predict_proba(x)
      # threshold the probabilities, otherwise get nans in the KL calculation
      probs = np.minimum(np.maximum(probs, 0.01), 0.99)
      return np.log(probs)  # return logits

    return enn_sampler

  return random_forest_agent
