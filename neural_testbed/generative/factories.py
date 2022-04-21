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

"""Convenient factory methods to help build generative models."""

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.generative import classification_envlikelihood as class_env


def make_2layer_mlp_logit_fn(
    input_dim: int,
    temperature: float,
    hidden: int,
    num_classes: int,
    key: chex.PRNGKey,
) -> class_env.LogitFn:
  """Factory method to create a generative model around a 2-layer MLP."""

  # Generating the logit function
  def net_fn(x: chex.Array) -> chex.Array:
    """Defining the generative model MLP."""
    y = hk.Linear(
        output_size=hidden,
        b_init=hk.initializers.RandomNormal(1./jnp.sqrt(input_dim)),
    )(x)
    y = jax.nn.relu(y)
    y = hk.Linear(hidden)(y)
    y = jax.nn.relu(y)
    return hk.Linear(num_classes)(y)

  transformed = hk.without_apply_rng(hk.transform(net_fn))
  dummy_input = jnp.zeros([1, input_dim])
  params = transformed.init(key, dummy_input)
  def forward(x: chex.Array) -> chex.Array:
    return transformed.apply(params, x) / temperature
  logit_fn = jax.jit(forward)

  return logit_fn


def make_filtered_gaussian_data(
    input_dim: int,
    logit_fn: class_env.LogitFn,
    reject_prob: float,
    fraction_rejected_classes: float,
    num_samples: int,
    key: chex.PRNGKey,
    max_itr: int = 30) -> testbed_base.Data:
  """Make a gaussian sampler that filters samples based on class labels."""
  # TODO(author2): WARNING - you cannot jit this function!
  def sample_gaussian_data(num_samples, key):
    data, _ = class_env.sample_gaussian_data(
        logit_fn=logit_fn,
        x_generator=class_env.make_gaussian_sampler(input_dim),
        num_train=num_samples,
        key=key,)
    return data

  rng = hk.PRNGSequence(key)
  dummy_logits = logit_fn(jnp.zeros([10, input_dim]))
  num_classes = dummy_logits.shape[1]
  num_rejected_classes = int(fraction_rejected_classes * num_classes)
  if num_rejected_classes == 0 or reject_prob == 0:
    return sample_gaussian_data(num_samples, next(rng))
  rejected_classes = jax.random.randint(
      next(rng), shape=(num_rejected_classes,), minval=0, maxval=num_classes)

  x_all = []
  y_all = []
  itr = 0
  total_samples = 0
  samples_per_itr = num_samples * 2

  while (total_samples < num_samples) and (itr < max_itr):
    data = sample_gaussian_data(samples_per_itr, next(rng))
    x, y = data.x, data.y

    mask_reject = jnp.isin(y.squeeze(), rejected_classes)
    uniform_probs = jax.random.uniform(next(rng), shape=(samples_per_itr,))
    mask_reject = mask_reject & (uniform_probs < reject_prob)
    x = x[~mask_reject]
    y = y[~mask_reject]
    x_all.append(x)
    y_all.append(y)
    itr += 1
    total_samples += jnp.sum(~mask_reject)

  if total_samples < num_samples:
    raise ValueError('Failed to sample required number of input data.')
  x_samples = jnp.concatenate(x_all)
  y_samples = jnp.concatenate(y_all)
  return testbed_base.Data(x_samples[:num_samples], y_samples[:num_samples])
