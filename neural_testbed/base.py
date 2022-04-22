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

"""Base classes for Neural testbed."""
import abc
import dataclasses
from typing import Any, Dict, NamedTuple, Optional

import chex
import typing_extensions


# Maybe this Data class needs to be a tf.Dataset
class Data(NamedTuple):
  x: chex.Array  # Always includes a batch index
  y: chex.Array  # Always includes a batch index


@dataclasses.dataclass(frozen=True)
class PriorKnowledge:
  """What an agent knows a priori about the problem."""
  input_dim: int
  num_train: int
  tau: int
  num_classes: int = 1
  layers: Optional[int] = None
  noise_std: Optional[float] = None
  temperature: Optional[float] = None
  extra: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ENNQuality:
  kl_estimate: float
  extra: Optional[Dict[str, Any]] = None


class EpistemicSampler(typing_extensions.Protocol):
  """Interface for drawing posterior samples from distribution.

  For classification this should represent the class *logits*.
  For regression this is the posterior sample of the function f(x).
  Assumes a batched input x.
  """

  def __call__(self, x: chex.Array, key: chex.PRNGKey) -> chex.Array:
    """Generate a random sample from approximate posterior distribution."""


class TestbedAgent(typing_extensions.Protocol):
  """An interface for specifying a testbed agent."""

  def __call__(self, data: Data, prior: PriorKnowledge) -> EpistemicSampler:
    """Sets up a training procedure given ENN prior knowledge."""


class TestbedProblem(abc.ABC):
  """An interface for specifying a generative GP model of data."""

  @abc.abstractproperty
  def train_data(self) -> Data:
    """Access training data from the GP for ENN training."""

  @abc.abstractproperty
  def prior_knowledge(self) -> PriorKnowledge:
    """Information describing the problem instance."""

  @abc.abstractmethod
  def evaluate_quality(self, enn_sampler: EpistemicSampler) -> ENNQuality:
    """Evaluate the quality of a posterior sampler."""

# See experiments/experiment.py for framework to run agent on a problem.

