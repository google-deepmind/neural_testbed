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

"""Utility functions for calculating likelihood."""

import abc
from typing import Tuple, Dict

import chex
import dataclasses
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
  # NOTE: This method should be pure.
  def test_data(self, seed: int) -> Tuple[testbed_base.Data, float]:
    """Generates a random sample of test data with posterior log-likelihood."""
    # TODO(author2): Note, no guarantees on format of test_data.


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


def gaussian_log_likelihood(err: chex.Array,
                            cov: chex.Array) -> float:
  """Calculates the Gaussian log likelihood of a multivariate normal."""
  first_term = len(err) * jnp.log(2 * jnp.pi)
  _, second_term = jnp.linalg.slogdet(cov)
  third_term = jnp.einsum('ai,ab,bi->i', err, jnp.linalg.pinv(cov), err)
  return -0.5 * (first_term + second_term + third_term)


def optimized_gaussian_ll(err: chex.Array) -> float:
  """Computes the Gaussian LL based on optimized residual MSE."""
  optimized_cov = jnp.mean(err ** 2) * jnp.eye(len(err))
  return gaussian_log_likelihood(err, optimized_cov)


def categorical_log_likelihood(class_probs: chex.Array,
                               data_labels: chex.Array) -> float:
  """Computes the categorical log likelihood of the data_labels."""
  num_data, unused_num_classes = class_probs.shape
  assert len(data_labels) == num_data
  assigned_probs = class_probs[jnp.arange(num_data), jnp.squeeze(data_labels)]
  return jnp.sum(jnp.log(assigned_probs))


def compute_discrete_kl(p: chex.Array, q: chex.Array) -> float:
  """KL-divergence between two discrete distributions with the same support."""
  # squeeze p and q if needed
  p = jnp.squeeze(p)
  q = jnp.squeeze(q)
  assert jnp.shape(p) == jnp.shape(q)

  return jnp.nansum(jnp.multiply(p, jnp.log(p) - jnp.log(q)))


@dataclasses.dataclass
class GaussianSampleKL(SampleBasedKL):
  """Evaluates KL according to optimized Gaussian residual model."""
  num_test_seeds: int
  num_enn_samples: int
  jit: bool = True

  def __call__(self,
               enn_sampler: testbed_base.EpistemicSampler,
               data_sampler: GenerativeDataSampler) -> testbed_base.ENNQuality:
    """Evaluates KL according to optimized Gaussian residual model."""
    batched_sampler = jax.vmap(enn_sampler, in_axes=[None, 0])
    batched_ll = jax.vmap(optimized_gaussian_ll)

    def kl_estimate(test_data_seed: int) -> float:
      """Computes KL estimate on a single instance of test data."""
      data, true_ll = data_sampler.test_data(test_data_seed)
      samples = batched_sampler(data.x, jnp.arange(self.num_enn_samples))
      batched_err = samples - jnp.expand_dims(data.y, 0)
      sampled_ll = batched_ll(batched_err)

      # TODO(author2): Make sure of our KL computation.
      unnormalized_kl = true_ll - average_sampled_log_likelihood(sampled_ll)
      return unnormalized_kl

    batched_kl = jax.vmap(kl_estimate)
    if self.jit:
      batched_kl = jax.jit(batched_kl)
    sampled_kl = batched_kl(jnp.arange(self.num_test_seeds))
    return testbed_base.ENNQuality(kl_estimate=jnp.mean(sampled_kl))


@dataclasses.dataclass
class CategoricalSampleKL(SampleBasedKL):
  """Evaluates KL according to categorical model."""
  num_test_seeds: int
  num_enn_samples: int
  jit: bool = True

  def __call__(self,
               enn_sampler: testbed_base.EpistemicSampler,
               data_sampler: GenerativeDataSampler) -> testbed_base.ENNQuality:
    """Evaluates KL according to categorical model."""
    batched_sampler = jax.vmap(enn_sampler, in_axes=[None, 0])
    batched_ll = jax.vmap(categorical_log_likelihood, in_axes=[0, None])

    def kl_estimate(test_data_seed: int) -> float:
      """Computes KL estimate on a single instance of test data."""
      data, true_ll = data_sampler.test_data(test_data_seed)
      enn_samples = batched_sampler(data.x, jnp.arange(self.num_enn_samples))
      class_probs = jax.nn.softmax(enn_samples)
      sampled_ll = batched_ll(class_probs, data.y)

      # TODO(author2): Make sure of our KL computation.
      unnormalized_kl = true_ll - average_sampled_log_likelihood(sampled_ll)
      return unnormalized_kl

    batched_kl = jax.vmap(kl_estimate)
    if self.jit:
      batched_kl = jax.jit(batched_kl)
    sampled_kl = batched_kl(jnp.arange(self.num_test_seeds))
    return testbed_base.ENNQuality(kl_estimate=jnp.mean(sampled_kl))


@dataclasses.dataclass
class ClassificationSampleAcc:
  """Evaluates accuracy for a classification problem."""
  num_test_seeds: int
  num_enn_samples: int
  jit: bool = True

  def __call__(self,
               enn_sampler: testbed_base.EpistemicSampler,
               data_sampler: GenerativeDataSampler) -> Dict[str, float]:
    """Evaluates accuracy for a classification problem."""
    batched_sampler = jax.vmap(enn_sampler, in_axes=[None, 0])

    def accuracy(data: testbed_base.Data):
      """Computes accuracy on the data."""
      batched_logits = batched_sampler(data.x, jnp.arange(self.num_enn_samples))
      batched_predictions = jnp.argmax(batched_logits, axis=2)
      chex.assert_shape(batched_predictions,
                        (self.num_enn_samples, data.y.shape[0]))
      return jax.numpy.mean(batched_predictions == jnp.transpose(data.y))

    def test_accuracy(test_data_seed: int) -> float:
      """Computes accuracy on a single batch of test data."""
      data, _ = data_sampler.test_data(seed=test_data_seed)
      return accuracy(data)

    def train_accuracy() -> float:
      """Computes accuracy on whole train data."""
      data = data_sampler.train_data
      return accuracy(data)

    # Train accuracy on whole train data
    sample_train_acc = train_accuracy()

    # Test accuracy on `self.num_test_seeds` batches of test data
    batched_test_acc = jax.vmap(test_accuracy)
    if self.jit:
      batched_test_acc = jax.jit(batched_test_acc)
    batched_sample_test_acc = batched_test_acc(jnp.arange(self.num_test_seeds))
    sample_test_acc = jnp.mean(batched_sample_test_acc)

    return {'train_acc': sample_train_acc, 'test_acc': sample_test_acc}


def add_classification_accuracy(
    sample_based_kl: SampleBasedKL, **kwargs) -> SampleBasedKL:
  """Adds classification accuracy to the metric evaluated by sample_based_kl."""
  default_kwargs = {'num_test_seeds': 10_000, 'num_enn_samples': 100}
  kwargs = {**default_kwargs, **kwargs}
  sample_based_acc = ClassificationSampleAcc(
      num_test_seeds=kwargs['num_test_seeds'],
      num_enn_samples=kwargs['num_enn_samples'])

  def evaluate_quality(
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: GenerativeDataSampler) -> testbed_base.ENNQuality:
    """Returns KL estimate and classification accuracy as ENN quality metrics."""
    enn_quality = sample_based_kl(enn_sampler, data_sampler)
    accuracy = sample_based_acc(enn_sampler, data_sampler)
    return testbed_base.ENNQuality(kl_estimate=enn_quality.kl_estimate,
                                   extra=accuracy)

  return evaluate_quality
