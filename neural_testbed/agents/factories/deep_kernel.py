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
from typing import Callable, Iterable, Optional, Sequence

import chex
from enn import base as enn_base
from enn import utils
import haiku as hk
import jax
import jax.numpy as jnp
from neural_testbed import base as testbed_base
from neural_testbed.agents.factories import base as factories_base
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
  num_train_steps: int = 1_000
  hidden_sizes: Sequence[int] = (50, 50)  # num_features is hidden_sizes[-1]
  batch_size: int = 100
  weight_decay: float = 1.0
  num_inference_samples: int = 32_768
  sigma_squared_factor: float = 1.0
  scale_factor: float = 25.0
  learning_rate: float = 1e-3
  dropout_rate: Optional[float] = None
  seed: int = 0  # Initialization seed
  normalization: int = Normalization.during_training


class MlpWithActivations(hk.Module):
  """A multi-layer perceptron module. Mostly copy-pasted from hk.nets.Mlp."""

  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      normalize_final: int = Normalization.during_training,
      name: Optional[str] = None,
  ):
    """Constructs an MLP where the last layer activation is available.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      normalize_final: How to normalize the activations of the penultimate
        layer.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError('When with_bias=False b_init must not be set.')

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    self.normalize_final = normalize_final
    layers = []
    for index, output_size in enumerate(output_sizes):
      layers.append(
          hk.Linear(
              output_size=output_size,
              w_init=w_init,
              b_init=b_init,
              with_bias=with_bias,
              name='linear_%d' % index))
    self.layers = tuple(layers)

  def __call__(
      self,
      inputs: jnp.ndarray,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jnp.ndarray:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError('When using dropout an rng key must be passed.')

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      # final layer:
      if i == (num_layers - 1):
        # normalize since we use this in inference
        if self.normalize_final == Normalization.during_training:
          out /= (1e-6 + jnp.expand_dims(jnp.linalg.norm(out, axis=1), 1))
          penultimate_out = out
        elif self.normalize_final == Normalization.only_output:
          penultimate_out = out
          penultimate_out /= (1e-6 +
                              jnp.expand_dims(jnp.linalg.norm(out, axis=1), 1))
        elif self.normalize_final == Normalization.no_normalization:
          penultimate_out = out
        else:
          raise ValueError('Invalid normalize_final setting')

      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return (out, penultimate_out)


def make_agent(config: DeepKernelConfig) -> testbed_base.TestbedAgent:
  """Factory method to create a deep kernel agent."""

  def deep_kernel_agent(
      data: testbed_base.Data,
      prior: testbed_base.PriorKnowledge,
  ) -> testbed_base.EpistemicSampler:
    """Output uniform class probabilities."""

    enn_data = enn_base.Batch(data.x, data.y)
    dataset = utils.make_batch_iterator(enn_data, config.batch_size,
                                        config.seed)

    def predict_fn(x):
      model = MlpWithActivations(
          output_sizes=list(config.hidden_sizes) + [prior.num_classes],
          activate_final=False,
          normalize_final=config.normalization)
      logits, final_layer_activations = model(
          x, dropout_rate=config.dropout_rate, rng=hk.next_rng_key())
      return (logits, final_layer_activations)

    def loss_fn(x, y):
      logits, _ = predict_fn(x)
      y_onehot = jax.nn.one_hot(jnp.squeeze(y, axis=-1), prior.num_classes)
      return jnp.mean(optax.softmax_cross_entropy(logits, y_onehot))

    rng = hk.PRNGSequence(config.seed)
    loss_fn_t = hk.transform(loss_fn)
    predict_fn_t = hk.transform(predict_fn)

    batch = next(dataset)
    params = loss_fn_t.init(next(rng), batch.x, batch.y)

    # use the same weight_decay heuristic as other agents
    weight_decay = config.weight_decay * jnp.sqrt(prior.temperature)
    optimizer = optax.adamw(config.learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    def train_step(params, opt_state, rng, x, y):
      grads = jax.grad(loss_fn_t.apply)(params, rng, x, y)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state

    for _ in range(config.num_train_steps):
      batch = next(dataset)
      params, opt_state = train_step(params, opt_state, next(rng), batch.x,
                                     batch.y)

    ##### prepare Cholesky factor #####

    # num_inference_samples controls how much training data to use for the
    # inference step, might run into memory issues if using all data
    num_inference_samples = min(config.num_inference_samples, data.x.shape[0])

    # B_train -> num_inference_samples
    d = utils.make_batch_iterator(enn_data, num_inference_samples, config.seed)
    # phi_x [B_train, num_features] (training data)
    _, phi_x = predict_fn_t.apply(params, next(rng), next(d).x)

    # at high temperature there is higher sampling noise
    sigma_squared = config.sigma_squared_factor * prior.temperature
    # [num_features, num_features]
    m = sigma_squared * jnp.eye(phi_x.shape[1]) + phi_x.T @ phi_x
    m_half = jax.scipy.linalg.cholesky(m, lower=True, overwrite_a=True)

    ##### Cholesky factor ready #####

    def enn_sampler(s: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
      # phi_s [B_test, num_features] (test data)
      rng = hk.PRNGSequence(key)
      mean_s, phi_s = predict_fn_t.apply(params, next(rng), s)

      # [num_features, num_classes]
      sample = jax.random.normal(
          next(rng), shape=(config.hidden_sizes[-1], prior.num_classes))
      # [num_features, num_classes]
      sample = jax.scipy.linalg.solve_triangular(
          m_half, sample, lower=True, trans=True, overwrite_b=True)

      scale = config.scale_factor * jnp.sqrt(sigma_squared)
      # [B_test, num_classes]
      sample = scale * phi_s @ sample

      # [B_test, num_classes]
      return mean_s + sample

    return enn_sampler

  return deep_kernel_agent


def deep_kernel_sweep() -> Sequence[DeepKernelConfig]:
  """Basic sweep over hyperparams."""
  sweep = []
  for scale_factor in [0., 1., 10., 100.]:
    for sigma_squared_factor in [1e-1, 1, 10]:
      for weight_decay in [1e-2, 1e-1, 1, 10, 100]:
        sweep.append(
            DeepKernelConfig(
                scale_factor=scale_factor,
                weight_decay=weight_decay,
                sigma_squared_factor=sigma_squared_factor))
  return tuple(sweep)


def paper_agent() -> factories_base.PaperAgent:
  return factories_base.PaperAgent(
      default=DeepKernelConfig(),
      ctor=make_agent,
      sweep=deep_kernel_sweep,
  )
