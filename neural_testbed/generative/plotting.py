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

"""Functions to sanity-check output in 1D plots."""

import chex
import jax
from neural_testbed import base as testbed_base
from neural_testbed import likelihood
import numpy as np
import pandas as pd
import plotnine as gg

# Setting plot theme
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing=0.5)


def sanity_1d(true_model: testbed_base.TestbedProblem,
              enn_sampler: testbed_base.EpistemicSampler,
              num_samples: int) -> gg.ggplot:
  """Sanity check to plot 1D representation of the GP testbed output."""
  x, y = true_model.train_data
  train_df = pd.DataFrame({'x': x[:, 0], 'y': y.ravel()})
  p = (gg.ggplot()
       + gg.geom_point(gg.aes(x='x', y='y'),
                       data=train_df, size=5, colour='blue'))

  if hasattr(true_model, 'problem'):
    true_model = true_model.problem  # Removing logging wrappers

  if not isinstance(true_model, likelihood.SampleBasedTestbed):
    return p

  categorical = true_model.prior_knowledge.num_classes > 1
  data, _ = true_model.data_sampler.test_data(0)
  sample_df = _gen_samples(enn_sampler, data.x, num_samples, categorical)
  p += gg.geom_line(gg.aes(x='x', y='y', group='factor(seed)'),
                    data=sample_df, alpha=0.5)
  if categorical:
    p += gg.geom_point(gg.aes(x='x', y='y'),
                       data=train_df[train_df.y == 0], size=5, colour='red')
    if hasattr(true_model, 'data_sampler'):
      # Add a green line y=probability of class 1
      fun_x = np.hstack([
          true_model.data_sampler.train_data.x[:, 0],
          true_model.data_sampler._x_test[:, 0],  # pylint:disable=protected-access
      ])
      fun_y = true_model.data_sampler.functions[0, :, 1]
      fun_df = pd.DataFrame({'x': fun_x, 'y': fun_y})
      p += gg.geom_line(gg.aes(x='x', y='y'), data=fun_df,
                        colour='green', size=1.5, alpha=0.5)
  elif hasattr(true_model, 'test_posterior'):
    gp_df = true_model.test_posterior
    p += gg.geom_line(gg.aes(x='x', y='mean'), data=gp_df, size=2, colour='red')
    p += gg.geom_ribbon(gg.aes(x='x', ymin='mean-2*std', ymax='mean+2*std'),
                        data=gp_df, fill='red', alpha=0.25)
  return p


def _gen_samples(enn_sampler: testbed_base.EpistemicSampler,
                 x: chex.Array,
                 num_samples: int,
                 categorical: bool = False) -> pd.DataFrame:
  """Generate posterior samples at x (not implemented for all posterior)."""
  data = []
  for seed in range(num_samples):
    net_out = enn_sampler(x, seed)
    y = jax.nn.softmax(net_out)[:, 1] if categorical else net_out[:, 0]
    data.append(pd.DataFrame({'x': x[:, 0], 'y': y, 'seed': seed}))
  return pd.concat(data)
