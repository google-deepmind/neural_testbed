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

"""Loading a leaderboard instance for the testbed."""

from typing import Callable, Tuple

import chex
import dataclasses
import jax
from neural_tangents.utils import typing as nt_types
from neural_testbed import base as testbed_base
from neural_testbed import generative
from neural_testbed import likelihood
from neural_testbed.leaderboard import sweep


KernelCtor = Callable[[int], nt_types.KernelFn]  # Maps input_dim -> KernelFn


@dataclasses.dataclass
class ClassificationTestbedConfig:
  """Configuration options for classification testbed instance."""
  num_train: int
  input_dim: int
  seed: int
  temperature: float
  num_classes: int = 2
  num_models: int = 10_000
  num_test_seeds: int = 1000
  num_enn_samples: int = 100
  kernel_ctor: KernelCtor = generative.make_benchmark_kernel
  num_layers: int = 1  # Output to prior knowledge


def gaussian_data(seed: int,
                  num_train: int,
                  input_dim: int,
                  num_test: int) -> Tuple[chex.Array, chex.Array]:
  """Generate Gaussian training and test data."""
  train_key, test_key = jax.random.split(jax.random.PRNGKey(seed))
  x_train = jax.random.normal(train_key, [num_train, input_dim])
  x_test = jax.random.normal(test_key, [num_test, input_dim])
  return x_train, x_test


def classification_load_from_config(
    config: ClassificationTestbedConfig) -> likelihood.SampleBasedTestbed:
  """Loads classification problem from config."""
  x_train, x_test = gaussian_data(
      seed=config.seed,
      num_train=config.num_train,
      input_dim=config.input_dim,
      num_test=config.num_test_seeds,
  )
  data_sampler = generative.GPClassificationEnsemble(
      kernel_fn=config.kernel_ctor(config.input_dim),
      x_train=x_train,
      x_test=x_test,
      num_classes=config.num_classes,
      temperature=config.temperature,
      num_models=config.num_models,
      seed=config.seed,
  )
  sample_based_kl = likelihood.CategoricalSampleKL(
      num_test_seeds=config.num_test_seeds,
      num_enn_samples=config.num_enn_samples,
  )
  sample_based_kl = likelihood.add_classification_accuracy(sample_based_kl)
  prior_knowledge = testbed_base.PriorKnowledge(
      input_dim=config.input_dim,
      num_train=config.num_train,
      num_classes=config.num_classes,
      layers=config.num_layers,
      temperature=config.temperature,
  )
  return likelihood.SampleBasedTestbed(
      data_sampler, sample_based_kl, prior_knowledge)


def classification_load(
    num_train: int, input_dim: int, seed: int, temperature: float,
) -> likelihood.SampleBasedTestbed:
  """Load classification GP from sweep hyperparameters."""
  config = ClassificationTestbedConfig(num_train, input_dim, seed, temperature)
  return classification_load_from_config(config)


def problem_from_id(gp_id: str) -> likelihood.SampleBasedTestbed:
  """Factory method to load leaderboard GP from gp_id."""
  if gp_id not in sweep.SETTINGS:
    raise ValueError(f'Unrecognised gp_id={gp_id}')
  elif 'classification' in gp_id:
    return classification_load(**sweep.SETTINGS[gp_id])
  else:
    raise ValueError(f'Load function unspecified for gp_id={gp_id}')
