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
"""Utilities for agent factories."""

from typing import Optional

import chex
from enn import base_legacy as enn_base
from enn import networks
from enn import supervised
from enn import utils
import jax
from neural_testbed import base as testbed_base


def extract_enn_sampler(
    experiment: supervised.BaseExperiment) -> testbed_base.EpistemicSampler:
  def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
    """Generate a random sample from posterior distribution at x."""
    net_out = experiment.predict(x, key)
    return networks.parse_net_output(net_out)
  return jax.jit(enn_sampler)


def make_iterator(data: testbed_base.Data,
                  batch_size: Optional[int] = None) -> enn_base.BatchIterator:
  batch = enn_base.Batch(data.x, data.y)
  return utils.make_batch_iterator(batch, batch_size)
