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
from enn import metrics
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.likelihood import base as likelihood_base
from neural_testbed.likelihood import utils


def compute_discrete_kl(p: chex.Array, q: chex.Array) -> float:
  """KL-divergence between two discrete distributions with the same support."""
  # squeeze p and q if needed
  p = jnp.squeeze(p)
  q = jnp.squeeze(q)
  assert jnp.shape(p) == jnp.shape(q)

  return jnp.nansum(jnp.multiply(p, jnp.log(p) - jnp.log(q)))


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
    kl_key, enn_key = jax.random.split(self.key, 2)
    test_data_fn = jax.jit(data_sampler.test_data)

    def get_logits(x: chex.Array) -> chex.Array:
      """Returns logits for input x."""
      sample_logits = functools.partial(enn_sampler, x)
      enn_keys = jax.random.split(enn_key, self.num_enn_samples)
      try:
        logits = jax.lax.map(sample_logits, enn_keys)
      except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
        # TODO(author1): replace with proper logging.
        print(f'Was not able to run enn_sampler inside jit due to: {e}')
        logits = jnp.array([sample_logits(k) for k in enn_keys])
      return logits

    def kl_estimator(key: chex.PRNGKey) -> float:
      """Computes KL estimate on a single instance of test data."""
      data, true_ll = test_data_fn(key)
      logits = get_logits(data.x)
      return true_ll - metrics.calculate_joint_ll(logits, data.y)

    kl_keys = jax.random.split(kl_key, self.num_test_seeds)

    # Attempt to use fully-jitted code, but if the enn_sampler is not able to
    # jax.jit then we fall back on another implementation.
    try:
      kl_estimates = jax.lax.map(kl_estimator, kl_keys)
    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
      # TODO(author1): replace with proper logging.
      print(f'Was not able to run enn_sampler inside jit due to: {e}')
      kl_estimates = jnp.array([kl_estimator(k) for k in kl_keys])

    return utils.parse_kl_estimates(kl_estimates)


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
    calculate_accuracy = metrics.make_accuracy_calculator()
    calculate_ece = metrics.SingleBatchECE(self.num_bins)

    def get_logits(x: chex.Array) -> chex.Array:
      """Returns logits for input x."""
      sample_logits = functools.partial(enn_sampler, x)
      enn_keys = jax.random.split(enn_key, self.num_enn_samples)
      try:
        logits = jax.lax.map(sample_logits, enn_keys)
      except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
        # TODO(author1): replace with proper logging.
        print(f'Was not able to run enn_sampler inside jit due to: {e}')
        logits = jnp.array([sample_logits(k) for k in enn_keys])
      return logits

    # We need all train data. We can get all the data by calling
    # data_sampler.train_data
    train_data = data_sampler.train_data
    train_data_logits = get_logits(train_data.x)
    train_acc = calculate_accuracy(train_data_logits, train_data.y)
    train_ece = calculate_ece(train_data_logits, train_data.y)

    # We need all test data. We can get a batch of the data by calling
    # data_sampler.test_data(key)
    # Keys for generating `self.num_test_seeds` batches of test data
    test_keys = jax.random.split(data_key, self.num_test_seeds)

    def test_x_y(key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
      data, _ = data_sampler.test_data(key)
      return (data.x, data.y)
    test_x, test_y = jax.lax.map(test_x_y, test_keys)
    # test_x has the shape of [num_test_seeds, tau] + single_data_shape. We
    # reshape it to [num_test_seeds * tau, ] + single_data_shape.
    test_x = jnp.reshape(test_x, (test_x.shape[0] * test_x.shape[1],) +
                         test_x.shape[2:])
    # test_y has the shape of [num_test_seeds, tau, 1]. We reshape it to
    # [num_test_seeds * tau, 1].
    test_y = jnp.reshape(test_y, (test_y.shape[0] * test_y.shape[1],) +
                         test_y.shape[2:])
    test_data = testbed_base.Data(x=test_x, y=test_y)
    test_data_logits = get_logits(test_data.x)
    test_acc = calculate_accuracy(test_data_logits, test_data.y)
    test_ece = calculate_ece(test_data_logits, test_data.y)

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_ece': train_ece,
        'test_ece': test_ece
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

