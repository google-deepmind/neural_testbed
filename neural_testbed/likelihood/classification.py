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

import dataclasses
import functools
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.likelihood import base as likelihood_base
from tensorflow_probability.substrates import jax as tfp


def compute_discrete_kl(p: chex.Array, q: chex.Array) -> float:
  """KL-divergence between two discrete distributions with the same support."""
  # squeeze p and q if needed
  p = jnp.squeeze(p)
  q = jnp.squeeze(q)
  assert jnp.shape(p) == jnp.shape(q)

  return jnp.nansum(jnp.multiply(p, jnp.log(p) - jnp.log(q)))


def categorical_log_likelihood(class_probs: chex.Array,
                               data_labels: chex.Array) -> float:
  """Computes the categorical log likelihood of the data_labels."""
  num_data, unused_num_classes = class_probs.shape
  assert len(data_labels) == num_data
  assigned_probs = class_probs[jnp.arange(num_data), jnp.squeeze(data_labels)]
  return jnp.sum(jnp.log(assigned_probs))


@jax.jit
def _avg_categorical_log_likelihood(class_logits: chex.Array,
                                    data_labels: chex.Array) -> float:
  """Computes the average categorical log likelihood of data_labels."""

  class_probs = jax.nn.softmax(class_logits)
  chex.assert_shape(
      class_probs,
      [class_logits.shape[0], data_labels.shape[0], class_logits.shape[-1]],
  )

  batched_ll = jax.vmap(categorical_log_likelihood, in_axes=[0, None])
  sampled_ll = batched_ll(class_probs, data_labels)
  ave_ll = likelihood_base.average_sampled_log_likelihood(sampled_ll)
  return ave_ll


@dataclasses.dataclass
class CategoricalKLSampledXSampledY(likelihood_base.SampleBasedKL):
  """Evaluates KL according to categorical model, sampling X and output Y.

  This approach samples an (x, y) output from the enn and data sampler and uses
  this to estimate the KL divergence.
  """
  num_test_seeds: int
  num_enn_samples: int
  key: chex.PRNGKey
  num_classes: Optional[int] = None  # Purely for shape checking.

  def __call__(
      self,
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: likelihood_base.GenerativeDataSampler,
  ) -> testbed_base.ENNQuality:
    """Evaluates KL according to categorical model."""

    kl_keys = jax.random.split(self.key, self.num_test_seeds)

    # Attempt to use fully-jitted code, but if the enn_sampler is not able to
    # jax.jit then we fall back on another implementation.
    try:
      def kl_estimator(key: chex.PRNGKey) -> float:
        """Computes KL estimate on a single instance of test data."""
        data_key, enn_key = jax.random.split(key)

        data, true_ll = data_sampler.test_data(data_key)

        enn_keys = jax.random.split(enn_key, self.num_enn_samples)

        logits = jax.lax.map(functools.partial(enn_sampler, data.x), enn_keys)
        avg_ll = _avg_categorical_log_likelihood(logits, data.y)
        return true_ll - avg_ll

      kl_estimates = jax.lax.map(kl_estimator, kl_keys)

    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
      # TODO(author1): replace with proper logging.
      print(f'Was not able to run enn_sampler inside jit due to: {e}')
      test_data_fn = jax.jit(data_sampler.test_data)
      def kl_estimator(key: chex.PRNGKey) -> float:
        """Computes KL estimate on a single instance of test data w/o jit."""

        data_key, enn_key = jax.random.split(key)
        enn_keys = jax.random.split(enn_key, self.num_enn_samples)
        data, true_ll = test_data_fn(data_key)
        logits = jnp.array([enn_sampler(data.x, k) for k in enn_keys])
        avg_ll = _avg_categorical_log_likelihood(logits, data.y)
        return true_ll - avg_ll
      kl_estimates = jnp.array([kl_estimator(k) for k in kl_keys])

    kl_estimate = jnp.mean(kl_estimates)
    extra = {
        'kl_estimate_std': float(jnp.std(kl_estimates)),
        'train_acc': None,
        'test_acc': None,
        'train_ece': None,
        'test_ece': None
    }
    return testbed_base.ENNQuality(kl_estimate=kl_estimate, extra=extra)


@dataclasses.dataclass
class ClassificationSampleAccEce:
  """Evaluates accuracy and expected calibration error (ece)."""
  num_test_seeds: int
  num_enn_samples: int
  key: chex.PRNGKey
  num_bins: int = 10  # Number of bins used in calculating ECE
  num_classes: Optional[int] = None  # Purely for shape checking.

  def __call__(
      self,
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: likelihood_base.GenerativeDataSampler,
  ) -> Dict[str, float]:
    """Evaluates accuracy and expected calibration error (ece)."""
    data_key, enn_key = jax.random.split(self.key)
    enn_keys = jax.random.split(enn_key, self.num_enn_samples)

    def enn_class_probabilities(data: testbed_base.Data) -> chex.Array:
      """Returns enn class probabilities for data."""

      def sample_enn(key: chex.PRNGKey) -> chex.Array:
        return enn_sampler(data.x, key)

      enn_samples = jax.lax.map(sample_enn, enn_keys)
      class_probs = jax.nn.softmax(enn_samples)
      chex.assert_shape(
          class_probs,
          [self.num_enn_samples, data.x.shape[0], self.num_classes],
      )
      mean_class_prob = jnp.mean(class_probs, axis=0)
      chex.assert_shape(mean_class_prob, [data.x.shape[0], self.num_classes])
      return mean_class_prob

    def accuracy_and_ece(data: testbed_base.Data) -> Tuple[float, float]:
      """Returns accuracy and expected calibration error (ece)."""
      probs = enn_class_probabilities(data)
      predictions = jnp.argmax(probs, axis=1)[:, None]
      chex.assert_shape(predictions, data.y.shape)

      # accuracy
      accuracy = jnp.mean(predictions == data.y)

      # ece
      num_data = len(data.x)
      logits = jnp.log(probs)
      chex.assert_shape(logits, (num_data, self.num_classes))
      labels_true = jnp.squeeze(data.y, axis=-1)
      chex.assert_shape(labels_true, (num_data,))
      labels_predicted = jnp.squeeze(predictions, axis=-1)
      chex.assert_shape(labels_predicted, (num_data,))
      ece = tfp.stats.expected_calibration_error(
          num_bins=self.num_bins,
          logits=logits,
          labels_true=labels_true,
          labels_predicted=labels_predicted)
      return accuracy, ece

    # We need all train data. We can get all the data by calling
    # data_sampler.train_data
    train_data = data_sampler.train_data
    train_acc, train_ece = accuracy_and_ece(train_data)

    # We need all test data. We can get a batch of the data by calling
    # data_sampler.test_data(key)
    # Keys for generating `self.num_test_seeds` batches of test data
    test_keys = jax.random.split(data_key, self.num_test_seeds)

    def test_x_y(key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
      data, _ = data_sampler.test_data(key)
      return (data.x, data.y)
    test_x, test_y = jax.lax.map(test_x_y, test_keys)
    # test_x has the shape of [num_test_seeds, tau, input_dim]. We reshape
    # it to [num_test_seeds * tau, input_dim]
    test_x = jax.numpy.reshape(test_x, (-1, test_x.shape[-1]))
    # test_y has the shape of [num_test_seeds, tau, 1]. We reshape
    # it to [num_test_seeds * tau, 1]
    test_y = jax.numpy.reshape(test_y, (-1, test_y.shape[-1]))
    test_data = testbed_base.Data(x=test_x, y=test_y)
    test_acc, test_ece = accuracy_and_ece(test_data)

    return {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'train_ece': float(train_ece),
        'test_ece': float(test_ece)
    }


def add_classification_accuracy_ece(
    sample_based_kl: likelihood_base.SampleBasedKL,
    num_test_seeds: int,
    num_enn_samples: int,
    num_bins: int = 10,
    num_classes: Optional[int] = None,
    **kwargs) -> likelihood_base.SampleBasedKL:
  """Adds classification accuracy to the metric evaluated by sample_based_kl."""
  del kwargs
  sample_based_acc_ece = ClassificationSampleAccEce(
      num_test_seeds=num_test_seeds,
      num_enn_samples=num_enn_samples,
      num_classes=num_classes,
      num_bins=num_bins,
      key=jax.random.PRNGKey(0)
  )

  def evaluate_quality(
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: likelihood_base.GenerativeDataSampler,
  ) -> testbed_base.ENNQuality:
    """Returns KL estimate and classification accuracy as ENN quality metrics."""
    enn_quality = sample_based_kl(enn_sampler, data_sampler)

    # Attempt to use jitted code, but if the enn_sampler is not able to
    # jax.jit then skip adding accuracy.
    try:
      eval_stats = sample_based_acc_ece(enn_sampler, data_sampler)
      enn_quality.extra.update(eval_stats)
    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
      print(f'Skipping accuracy. The enn_sampler not jittable due to \n{e}')
    return testbed_base.ENNQuality(
        kl_estimate=enn_quality.kl_estimate, extra=enn_quality.extra)

  return evaluate_quality

