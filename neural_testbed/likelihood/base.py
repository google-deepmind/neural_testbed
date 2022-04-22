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

"""Utility functions for calculating likelihood."""

import abc
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
import typing_extensions


class GenerativeDataSampler(abc.ABC):
  """An interface for generative test/train data."""

  @abc.abstractproperty
  def train_data(self) -> testbed_base.Data:
    """Access training data from the GP for ENN training."""

  @abc.abstractmethod
  def test_data(self, key: chex.PRNGKey) -> Tuple[testbed_base.Data, float]:
    """Generates a random sample of test data with posterior log-likelihood.

    WARNING: This method should be pure, for use in jax.jit.

    Args:
      key: random number generator key.
    """


class SampleBasedKL(typing_extensions.Protocol):
  """An interface for estimating KL divergence from data."""

  def __call__(self,
               enn_sampler: testbed_base.EpistemicSampler,
               data_sampler: GenerativeDataSampler) -> testbed_base.ENNQuality:
    """Uses ENN and data samples to estimate the KL divergence."""


class SampleBasedTestbed(testbed_base.TestbedProblem):
  """A simpler wrapper to make a testbed instance out of data sampler."""

  def __init__(self,
               data_sampler: GenerativeDataSampler,
               sample_based_kl: SampleBasedKL,
               prior_knowledge: testbed_base.PriorKnowledge):
    self.data_sampler = data_sampler
    self.sample_based_kl = sample_based_kl
    self._prior_knowledge = prior_knowledge

  @property
  def train_data(self) -> testbed_base.Data:
    return self.data_sampler.train_data

  def evaluate_quality(
      self,
      enn_sampler: testbed_base.EpistemicSampler) -> testbed_base.ENNQuality:
    """Evaluate the quality of a posterior sampler."""
    return self.sample_based_kl(enn_sampler, self.data_sampler)

  @property
  def prior_knowledge(self) -> testbed_base.PriorKnowledge:
    return self._prior_knowledge


def average_sampled_log_likelihood(x: chex.Array) -> float:
  """Computes average log likelihood from samples.

  This method takes several samples of log-likelihood, converts
  them to likelihood (by exp), then takes the average, then
  returns the logarithm over the average  LogSumExp
  trick is used for numerical stability.

  Args:
    x: chex.Array
  Returns:
    log-mean-exponential
  """
  return jax.lax.cond(
      jnp.isneginf(jnp.max(x)),
      lambda x: -jnp.inf,
      lambda x: jnp.log(jnp.mean(jnp.exp(x - jnp.max(x)))) + jnp.max(x),
      operand=x,
  )


class MetricCalculator(typing_extensions.Protocol):
  """Interface for evaluation of multiple posterior samples based on a metric."""

  def __call__(self, logits: chex.Array, labels: chex.Array) -> float:
    """Calculates a metric based on logits and labels.

    Args:
      logits: An array of shape [A, B, C] where B is the batch size of data, C
        is the number of outputs per data (for classification, this is
        equal to number of classes), and A is the number of random samples for
        each data.
      labels: An array of shape [B, 1] where B is the batch size of data.

    Returns:
      A float number specifies the value of the metric.
    """
