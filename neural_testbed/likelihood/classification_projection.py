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

"""Calculating classification likelihood based on random projection.

WARNING: THIS IS EXPERIMENTAL CODE AND NOT YET AT GOLD QUALITY.
"""
# TODO(author2): sort out the code quality here.

import dataclasses
import functools
from typing import Tuple

import chex
from enn.extra import kmeans
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from neural_testbed import base as testbed_base
from neural_testbed.likelihood import base as likelihood_base
from neural_testbed.likelihood import utils
import typing_extensions


_Counts = chex.Array  # Number of elements in each cluster: [num_clusters]
_Centers = chex.Array  # The center of each cluser: [num_clusters, tau]


class ClusterAlg(typing_extensions.Protocol):

  def __call__(self,
               probits: chex.Array,
               y: chex.Array,
               key: chex.PRNGKey) -> Tuple[_Counts, _Centers]:
    """Uses probits and y labels to compute counts and centers."""


@dataclasses.dataclass
class JointLLCalculatorProjection(likelihood_base.MetricCalculator):
  """Computes joint ll aggregated over enn samples using projection method.

  Depending on data batch_size (can be inferred from logits and labels), this
  function computes joint ll for tau=batch_size aggregated over enn samples. If
  data batch_size is one, this function computes marginal ll.
  """
  cluster_alg: ClusterAlg  # Algorithm for clustering
  clip_probits: float  # Clip absolute value of probits at this level
  cluster_key: chex.PRNGKey  # An RNG key

  def __call__(self, logits: chex.Array, labels: chex.Array) -> float:
    """Computes joint ll aggregated over enn samples using projection method."""
    num_enn_samples, tau, num_classes = logits.shape
    chex.assert_shape(labels, [tau, 1])

    def logits_to_probits(logits: chex.Array) -> chex.Array:
      probs = jax.nn.softmax(logits)
      probits = jsp.ndtri(probs)
      probits = jnp.clip(probits, -self.clip_probits, self.clip_probits)
      chex.assert_shape(probits, [tau, num_classes])
      return probits

    # Convert logits to probits
    probits = jax.lax.map(logits_to_probits, logits)

    # Perform appropriate clustering
    counts, centers = self.cluster_alg(probits, labels, self.cluster_key)
    chex.assert_rank([counts, centers], [1, 2])
    chex.assert_shape(centers, [counts.shape[0], tau])

    # Compute the model log likelihood
    log_sum_prod = jax.jit(utils.log_sum_prod)
    avg_ll = log_sum_prod(counts / num_enn_samples, centers)
    return avg_ll


@dataclasses.dataclass
class CategoricalClusterKL(likelihood_base.SampleBasedKL):
  """Evaluates KL according to random clustering of ENN samples."""
  cluster_alg: ClusterAlg  # Algorithm for clustering
  num_enn_samples: int  # Number of samples from ENN environment model
  num_test_seeds: int  # Number of testing seeds for the data generation
  key: chex.PRNGKey  # RNG key
  clip_probits: float = 5  # Clip absolute value of probits at this level

  def __call__(
      self,
      enn_sampler: testbed_base.EpistemicSampler,
      data_sampler: likelihood_base.GenerativeDataSampler,
  ) -> testbed_base.ENNQuality:
    """Evaluates KL according to categorical model."""
    kl_key, enn_key, cluster_key = jax.random.split(self.key, 3)
    joint_ll_calculator = JointLLCalculatorProjection(
        self.cluster_alg,
        self.clip_probits,
        cluster_key)
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

    def kl_estimate(key: chex.PRNGKey) -> chex.Array:
      """Estimates the KL for one realization of the data."""
      data, true_ll = test_data_fn(key)
      logits = get_logits(data.x)
      return true_ll - joint_ll_calculator(logits, data.y)

    kl_keys = jax.random.split(kl_key, self.num_test_seeds)
    try:
      kl_estimates = jax.lax.map(kl_estimate, kl_keys)
    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError) as e:
      # TODO(author1): replace with proper logging.
      print(f'Was not able to run kl_estimate inside jit due to: {e}')
      kl_estimates = jnp.array([kl_estimate(k) for k in kl_keys])

    return utils.parse_kl_estimates(kl_estimates)


@dataclasses.dataclass
class KmeansCluster(ClusterAlg):
  """Clusters probits based on K-Means algorithm."""
  num_centroids: int  # Number of KMeans centroids
  max_iter: int = 10  # Number of iterations of KMeans
  cluster_only_correct_class: bool = False  # Only cluster on correct class

  def __call__(self,
               probits: chex.Array,
               y: chex.Array,
               key: chex.PRNGKey) -> Tuple[_Counts, _Centers]:
    """Cluster ENN probits according to K-Means."""
    chex.assert_rank(probits, 3)
    num_enn_samples, tau, num_classes = probits.shape

    if self.cluster_only_correct_class:
      flat_probits = probits[:, jnp.arange(tau), y.ravel()]
      chex.assert_shape(flat_probits, [num_enn_samples, tau])
    else:
      flat_probits = jnp.reshape(probits, [num_enn_samples, tau * num_classes])

    # Fit the KMeans clustering algorithm
    kmeans_cluster = kmeans.KMeansCluster(
        self.num_centroids, self.max_iter, key)
    output = kmeans_cluster.fit(flat_probits)

    # Parse the output and reshape.
    counts = output.counts_per_centroid
    chex.assert_shape(counts, [self.num_centroids])

    # obtain classes from the output and convert to one-hot encoding
    classes = jax.nn.one_hot(output.classes, self.num_centroids)
    chex.assert_shape(classes, [num_enn_samples, self.num_centroids])

    # get probs_correct
    if self.cluster_only_correct_class:
      probits_correct = flat_probits
    else:
      probits_correct = probits[:, jnp.arange(tau), y.ravel()]
      chex.assert_shape(probits_correct, [num_enn_samples, tau])

    probs_correct = jsp.ndtr(probits_correct)

    # unnormalized prob_correct_centers
    probs_correct_centers = jnp.einsum('ij,ik->jk', classes, probs_correct)
    chex.assert_shape(probs_correct_centers, [self.num_centroids, tau])

    # normalize prob_correct_centers
    counts_denominator = counts[:, None]
    chex.assert_shape(counts_denominator, [self.num_centroids, 1])

    probs_correct_centers = jnp.divide(probs_correct_centers + 5e-10,
                                       counts_denominator + 1e-9)

    chex.assert_shape(probs_correct_centers, [self.num_centroids, tau])

    return counts, probs_correct_centers


@dataclasses.dataclass
class RandomProjection(ClusterAlg):
  """Cluster ENN probits according to random projections."""
  dimension: int
  cluster_only_correct_class: bool = False  # Only cluster on correct class
  normalize: bool = False  # Whether to apply per-class normalization

  def __call__(self,
               probits: chex.Array,
               y: chex.Array,
               key: chex.PRNGKey) -> Tuple[_Counts, _Centers]:

    def cluster_fn(probits: chex.Array,
                   y: chex.Array,
                   key: chex.PRNGKey) -> Tuple[_Counts, _Centers]:
      chex.assert_rank(probits, 3)
      num_enn_samples, tau, num_classes = probits.shape
      if self.cluster_only_correct_class:
        flat_probits = probits[:, jnp.arange(tau), y.ravel()]
        target_shape = [num_enn_samples, tau]
      else:
        flat_probits = jnp.reshape(
            probits, [num_enn_samples, tau * num_classes])
        target_shape = [num_enn_samples, tau * num_classes]

      if self.normalize:
        flat_probits -= jnp.mean(flat_probits, axis=0, keepdims=True)
        svd_u, _, svd_v = jnp.linalg.svd(flat_probits, full_matrices=False)
        flat_probits = jnp.dot(svd_u, svd_v)

      a_key, b_key = jax.random.split(key)
      chex.assert_shape(flat_probits, target_shape)

      # Compute the correct label probabilities
      probs = jsp.ndtr(probits)
      select_label = jax.jit(jax.vmap(_select_label, in_axes=[0, None]))
      probs_correct = select_label(probs, y)
      chex.assert_shape(probs_correct, [num_enn_samples, tau])

      # Perfom the projection
      if self.cluster_only_correct_class:
        a = jax.random.normal(a_key, shape=[self.dimension, tau])
      else:
        a = jax.random.normal(a_key, shape=[self.dimension, tau * num_classes])
      b = jax.random.normal(b_key, shape=[self.dimension])

      def compute_projection(flat_probit: chex.Array) -> chex.Array:
        projection = jnp.dot(a, flat_probit) + b
        return (projection > 0).astype(jnp.int32)

      projections = jax.vmap(compute_projection)(flat_probits)
      chex.assert_shape(projections, [num_enn_samples, self.dimension])
      batched_equal = jax.vmap(jnp.array_equal, in_axes=(0, None))

      num_features = min(num_enn_samples, 2**self.dimension)

      # Choose features as the unique projections, and also get counts
      features, counts = jnp.unique(projections, axis=0,
                                    size=num_features,
                                    return_counts=True)
      chex.assert_shape(features, [num_features, self.dimension])
      chex.assert_shape(counts, [num_features])

      # TODO(author2): Consider simplifying this step
      # Compute the average for each feature
      def single_center_per_t(single_f: chex.Array, t: int) -> chex.Array:
        # Check for the rows where samples match f
        valid_masks = batched_equal(
            projections, single_f).astype(jnp.float32)
        chex.assert_shape(valid_masks, [num_enn_samples])

        numerator = jnp.sum(valid_masks * probs_correct[:, t])
        denominator = jnp.sum(valid_masks)
        return (numerator + 1e-6) / (denominator + 1e-6)

      single_center = jax.vmap(single_center_per_t, in_axes=[None, 0])
      map_fn = lambda x: single_center(x, jnp.arange(tau))
      all_centers = jax.lax.map(map_fn, features)
      chex.assert_shape(all_centers, [num_features, tau])

      return counts, all_centers

    return jax.jit(cluster_fn)(probits, y, key)


def _select_label(probs: chex.Array, y: chex.Array) -> chex.Array:
  chex.assert_rank(probs, 2)
  labels = jnp.squeeze(y, axis=1)
  chex.assert_rank(labels, 1)
  return probs[jnp.arange(probs.shape[0]), labels]
