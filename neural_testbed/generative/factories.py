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

"""Convenient factory methods to help build generative models."""

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed.generative import classification_envlikelihood


def make_2layer_mlp_generative_model(
    input_dim: int,
    num_train: int,
    key: chex.PRNGKey,
    temperature: float,
    tau: int,
    hidden: int,
    num_classes: int,
) -> classification_envlikelihood.ClassificationEnvLikelihood:
  """Factory method to create a generative model around a 2-layer MLP."""
  rng = hk.PRNGSequence(key)

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
  params = transformed.init(next(rng), dummy_input)
  def forward(x: chex.Array) -> chex.Array:
    return transformed.apply(params, x) / temperature

  # Generating the Gaussian data generator
  def x_generator(k: chex.PRNGKey, num_samples: int) -> chex.Array:
    return jax.random.normal(k, [num_samples, input_dim])

  return classification_envlikelihood.ClassificationEnvLikelihood(
      logit_fn=jax.jit(forward),
      x_generator=x_generator,
      num_train=num_train,
      key=next(rng),
      tau=tau,
  )
