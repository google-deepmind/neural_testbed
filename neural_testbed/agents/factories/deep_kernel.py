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
"""Factory methods for deep kernel agent.

This works as follows:
  1) Train a deep neural network on the data as normal (with regularization etc
     if required).
  2) Treat the learned mapping in the network as a *deep kernel* that is taking
     the input data and mapping it to a new space where the examples are
     linearly separable.
  3) During inference we treat the learned deep kernel as the kernel in a
     Gaussian process.
  4) We do some clever linear algebra to keep the inference (relatively)
     tractable as the problem size and number of data increases.

"""

import dataclasses
import functools
from typing import Callable, Iterable, NamedTuple, Optional, Sequence

import chex
from enn import datasets
from enn import losses
from enn import utils
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
import optax


@dataclasses.dataclass
class Normalization:
  """Enum categorizing how we normalize the activations in the last layer."""
  during_training = 1
  only_output = 2
  no_normalization = 3


@dataclasses.dataclass
class DeepKernelConfig:
  """Deep kernel config."""
  num_train_steps: int = 1_000  # number of training steps
  batch_size: int = 100  # batch size to train with
  learning_rate: float = 1e-3  # training learning rate
  weight_decay: float = 1.0  # l2 weight decay
  hidden_sizes: Sequence[int] = (50, 50)  # num_features is hidden_sizes[-1]
  scale_factor: float = 2.0  # sampling scale factor
  num_inference_samples: int = 32_768  # max number of train data to use
  sigma_squared_factor: float = 4.0  # noise factor
  seed: int = 0  # initialization seed
  normalization: int = Normalization.only_output  #  how to normalize last layer


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState


class MlpWithActivations(hk.Module):
  """A multi-layer perceptron module. Mostly copy-pasted from hk.nets.Mlp."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      normalize_final: int = Normalization.during_training,
      name: Optional[str] = None,
  ):
    """Constructs an MLP where the last layer activation is available.

    Args:
      output_sizes: Sequence of layer sizes.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      normalize_final: How to normalize the activations of the penultimate
        layer.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """

    super().__init__(name=name)
    self.activation = activation
    self.normalize_final = normalize_final
    layers = []
    for index, output_size in enumerate(output_sizes):
      layers.append(
          hk.Linear(output_size=output_size, name='linear_%d' % index))
    self.layers = tuple(layers)

  def __call__(
      self,
      inputs: jnp.ndarray,
  ) -> jnp.ndarray:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    num_layers = len(self.layers)
    out = hk.Flatten()(inputs)
    for i, layer in enumerate(self.layers):
      if i == (num_layers - 1):  # this is the final layer, apply normalization:
        if self.normalize_final == Normalization.during_training:
          out /= (1e-6 + jnp.expand_dims(jnp.linalg.norm(out, axis=1), 1))
          penultimate_out = out
        elif self.normalize_final == Normalization.only_output:
          penultimate_out = out / (
              1e-6 + jnp.expand_dims(jnp.linalg.norm(out, axis=1), 1))
        elif self.normalize_final == Normalization.no_normalization:
          penultimate_out = out
        else:
          raise ValueError('Invalid normalize_final setting')

      out = layer(out)
      if i < (num_layers - 1):  # don't activate final layer
        out = self.activation(out)

    return (out, penultimate_out)  # pytype: disable=bad-return-type  # jax-ndarray


def make_agent(config: DeepKernelConfig) -> testbed_base.TestbedAgent:
  """Factory method to create a deep kernel agent."""

  def deep_kernel_agent(
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge,
  ) -> testbed_base.EpistemicSampler:
    """Output uniform class probabilities."""
    rng = hk.PRNGSequence(config.seed)

    enn_data = datasets.ArrayBatch(x=data.x, y=data.y)
    dataset = utils.make_batch_iterator(enn_data, config.batch_size,
                                        config.seed)

    def predict_fn(x):
      model = MlpWithActivations(
          output_sizes=list(config.hidden_sizes) + [prior.num_classes],
          normalize_final=config.normalization)
      logits, final_layer_activations = model(x)
      return (logits, final_layer_activations)

    predict_fn_t = hk.without_apply_rng(hk.transform(predict_fn))
    params = predict_fn_t.init(next(rng), next(dataset).x)

    # helper function to conform to testbed api
    def net(params, x, index):
      del index
      logits, _ = predict_fn_t.apply(params, x)
      return logits

    # use the same weight_decay heuristic as other agents
    weight_decay = (
        config.weight_decay * jnp.sqrt(prior.temperature) * prior.input_dim /
        prior.num_train)

    single_loss = losses.combine_single_index_losses_as_metric(
        # This is the loss you are training on.
        train_loss=losses.XentLoss(prior.num_classes),
        # We will also log the accuracy in classification.
        extra_losses={
            'acc': losses.AccuracyErrorLoss(prior.num_classes)
        },
    )
    single_loss = losses.wrap_single_loss_as_single_loss_no_state(single_loss)
    loss_fn = losses.average_single_index_loss_no_state(single_loss,)
    loss_fn = losses.add_l2_weight_decay_no_state(loss_fn, scale=weight_decay)  # pytype: disable=wrong-arg-types  # jax-types
    loss_fn = jax.jit(functools.partial(loss_fn, net))

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    def train_step(state, batch):
      _, grads = jax.value_and_grad(
          loss_fn, has_aux=True)(state.params, batch, None)
      updates, new_opt_state = optimizer.update(grads, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return TrainingState(new_params, new_opt_state)

    state = TrainingState(params, opt_state)
    for _ in range(config.num_train_steps):
      batch = next(dataset)
      state = train_step(state, batch)

    ##### prepare Cholesky factor #####

    # num_inference_samples controls how much training data to use for the
    # inference step, might run into memory issues if using all data
    num_inference_samples = min(config.num_inference_samples, data.x.shape[0])

    # B_train -> num_inference_samples
    d = utils.make_batch_iterator(enn_data, num_inference_samples, config.seed)
    # phi_x [B_train, num_features] (training data)
    _, phi_x = predict_fn_t.apply(state.params, next(d).x)

    # at high temperature there is higher sampling noise
    sigma_squared = config.sigma_squared_factor * prior.temperature
    # [num_features, num_features]
    m = sigma_squared * jnp.eye(phi_x.shape[1]) + phi_x.T @ phi_x
    m_half = jax.scipy.linalg.cholesky(m, lower=True, overwrite_a=True)

    ##### Cholesky factor ready #####

    def enn_sampler(x: chex.Array, key: chex.PRNGKey) -> chex.Array:
      # phi_s [B_test, num_features] (test data)
      rng = hk.PRNGSequence(key)
      mean_s, phi_s = predict_fn_t.apply(state.params, x)

      # [num_features, num_classes]
      sample = jax.random.normal(
          next(rng), shape=(config.hidden_sizes[-1], prior.num_classes))
      # [num_features, num_classes]
      sample = jax.scipy.linalg.solve_triangular(
          m_half, sample, lower=True, trans=True, overwrite_b=True)

      scale = (
          config.scale_factor * jnp.sqrt(sigma_squared) /
          jnp.sqrt(prior.num_train) / jnp.sqrt(prior.temperature))
      # [B_test, num_classes]
      return mean_s + scale * phi_s @ sample  # sampled logit from posterior

    return enn_sampler

  return deep_kernel_agent
