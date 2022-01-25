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

from typing import Dict

import chex
import haiku as hk
import jax
from neural_tangents.utils import typing as nt_types
from neural_testbed import base as testbed_base
from neural_testbed import likelihood
from neural_testbed.generative import classification_envlikelihood
from neural_testbed.generative import gp_regression
from neural_testbed.generative import nt_kernels
import numpy as np
import pandas as pd
import plotnine as gg


def set_gg_theme():
  """Sets the global ggplot theme."""
  try:
    # TODO(author2): Understand why this is causing errors in testing.
    gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
    gg.theme_update(figure_size=(12, 8), panel_spacing=0.5)
  except RuntimeError:
    pass


def sanity_plots(
    true_model: testbed_base.TestbedProblem,
    enn_sampler: testbed_base.EpistemicSampler,
) -> Dict[str, gg.ggplot]:
  """Sanity check plots for output of GP testbed output."""
  set_gg_theme()
  if hasattr(true_model, 'problem'):
    true_model = true_model.problem  # Removing logging wrappers
  prior = true_model.prior_knowledge

  # Specialized plotting for the 2D classification infra.
  if prior.num_classes == 2 and prior.input_dim == 2:
    # TODO(author2): annotate true_model as classification.
    if not hasattr(true_model, 'data_sampler'):
      raise ValueError('Error in plotting infrastructure.')
    problem = true_model.data_sampler
    return generate_2d_plots(problem, enn_sampler)  # pytype:disable=wrong-arg-types
  else:
    return {'enn': sanity_1d(true_model, enn_sampler)}


def sanity_1d(true_model: testbed_base.TestbedProblem,
              enn_sampler: testbed_base.EpistemicSampler) -> gg.ggplot:
  """Sanity check to plot 1D representation of the GP testbed output."""
  set_gg_theme()
  if hasattr(true_model, 'problem'):
    true_model = true_model.problem  # Removing logging wrappers
  if not hasattr(true_model, 'data_sampler'):
    return gg.ggplot()
  if true_model.prior_knowledge.num_classes == 1:
    gp_model = true_model.data_sampler
    if not isinstance(gp_model, gp_regression.GPRegression):
      print('WARNING: no plot implemented')
      return gg.ggplot()
    return plot_1d_regression(gp_model, enn_sampler)
  else:
    if not isinstance(true_model, likelihood.SampleBasedTestbed):
      raise ValueError('Unrecognised testbed for classification plot.')
    return plot_1d_classification(true_model, enn_sampler)


def _gen_samples(enn_sampler: testbed_base.EpistemicSampler,
                 x: chex.Array,
                 num_samples: int,
                 categorical: bool = False) -> pd.DataFrame:
  """Generate posterior samples at x (not implemented for all posterior)."""
  # Generate the samples
  data = []
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed=0))
  for seed in range(num_samples):
    net_out = enn_sampler(x, next(rng))
    y = jax.nn.softmax(net_out)[:, 1] if categorical else net_out[:, 0]
    data.append(pd.DataFrame({'x': x[:, 0], 'y': y, 'seed': seed}))
  sample_df = pd.concat(data)

  # Aggregate the samples for plotting
  def pct_95(x):
    return np.percentile(x, 95)
  def pct_5(x):
    return np.percentile(x, 5)
  enn_df = (sample_df.groupby('x')['y']
            .agg([np.mean, np.std, pct_5, pct_95]).reset_index())
  enn_df = enn_df.rename({'mean': 'y'}, axis=1)
  enn_df['method'] = 'enn'
  return enn_df


def plot_1d_regression(gp_model: gp_regression.GPRegression,
                       enn_sampler: testbed_base.EpistemicSampler,
                       num_samples: int = 100) -> gg.ggplot:
  """Plots 1D regression with confidence intervals."""
  # Training data
  train_data = gp_model.train_data
  df = pd.DataFrame({'x': train_data.x[:, 0], 'y': train_data.y[:, 0]})
  # Posterior data
  posterior_df = pd.DataFrame({
      'x': gp_model.x_test[:, 0],
      'y': gp_model.test_mean[:, 0],
      'std': np.sqrt(np.diag(gp_model.test_cov)),
  })
  posterior_df['method'] = 'gp'
  # ENN data
  enn_df = _gen_samples(enn_sampler, gp_model.x_test, num_samples)
  p = (gg.ggplot(pd.concat([posterior_df, enn_df]))
       + gg.aes(x='x', y='y', ymin='y-std', ymax='y+std', group='method')
       + gg.geom_ribbon(gg.aes(fill='method'), alpha=0.25)
       + gg.geom_line(gg.aes(colour='method'), size=2)
       + gg.geom_point(gg.aes(x='x', y='y'), data=df, size=4, inherit_aes=False)
       + gg.scale_colour_manual(['#e41a1c', '#377eb8'])
       + gg.scale_fill_manual(['#e41a1c', '#377eb8'])
      )
  return p


def plot_1d_classification(true_model,  # TODO(author2): add typing
                           enn_sampler: testbed_base.EpistemicSampler,
                           num_samples: int = 100) -> gg.ggplot:
  """Plots 1D classification with ENN samples."""
  x, y = true_model.train_data

  # Pulling out the training data
  df = pd.DataFrame({'x': x[:, 0], 'y': y[:, 0]})

  # Generate samples from the ENN at 1000 randomly generated test datapoints.
  def gen_test(key: chex.PRNGKey) -> testbed_base.Data:
    data, _ = true_model.data_sampler.test_data(key)
    return testbed_base.Data(x=data.x[0, :], y=data.y[0, :])
  data_keys = jax.random.split(jax.random.PRNGKey(seed=0), 1000)
  data = jax.jit(jax.vmap(gen_test))(data_keys)
  enn_df = _gen_samples(enn_sampler, data.x, num_samples, categorical=True)

  # Calculate the true function distribution
  x = true_model.data_sampler.test_x
  _, input_dim = x.shape
  prob_df = pd.DataFrame({
      'x': x[:, 0],
      'y': true_model.data_sampler.probabilities[:, 1],
  })
  prob_df['std'] = 0
  prob_df['method'] = 'true_function'

  p = (gg.ggplot(pd.concat([prob_df, enn_df]))
       + gg.aes(x='x', y='y', ymin='pct_5', ymax='pct_95', group='method')
       + gg.geom_hline(yintercept=0, alpha=0.2, linetype='dashed')
       + gg.geom_hline(yintercept=1, alpha=0.2, linetype='dashed')
       + gg.geom_ribbon(gg.aes(fill='method'), alpha=0.25)
       + gg.geom_line(gg.aes(colour='method'), size=2)
       + gg.geom_point(gg.aes(x='x', y='y'), data=df[df.y == 1],
                       size=5, colour='#377eb8', inherit_aes=False)
       + gg.geom_point(gg.aes(x='x', y='y'), data=df[df.y == 0],
                       size=5, colour='#e41a1c', inherit_aes=False)
       + gg.scale_colour_manual(['green', 'black'])
       + gg.scale_fill_manual(['green', 'black'])
       + gg.ylab('probability of class 1')
       + gg.xlab(f'x[0] of {input_dim}-dimensional input.')
      )
  return p


def investigate_1d_regression_model(
    kernel_fn: nt_types.KernelFn = nt_kernels.make_benchmark_kernel(),
    num_train: int = 5) -> gg.ggplot:
  """Plots the 1D posterior for random training data in regression model.

  This plot is effectively a poor-man's test... just to be able to visually
  inspect the qualitative behaviour of the 1D regression posterior.

  Args:
    kernel_fn: kernel function defining the GP.
    num_train: number of training points.

  Returns:
    gg.ggplot investigation of 1D posterior.
  """
  x_test = np.random.randn(1000, 1)
  data_sampler = gp_regression.GPRegression(
      kernel_fn,
      x_train=np.random.randn(num_train, 1),
      x_test=x_test,
      key=jax.random.PRNGKey(13),
      noise_std=0.3,
      tau=100,
  )
  train_data = data_sampler.train_data
  df = pd.DataFrame({'x': train_data.x[:, 0], 'y': train_data.y[:, 0]})
  plt_df = pd.DataFrame({
      'x': x_test[:, 0],
      'mean': data_sampler._test_mean[:, 0],  # pylint:disable=protected-access
      'std': np.sqrt(np.diag(data_sampler._test_cov)),  # pylint:disable=protected-access
  })

  p = (gg.ggplot(plt_df)
       + gg.aes(x='x')
       + gg.geom_line(gg.aes(y='mean', ymin='mean-std', ymax='mean+std'),
                      colour='red', size=2)
       + gg.geom_ribbon(gg.aes(y='mean', ymin='mean-std', ymax='mean+std'),
                        alpha=0.25, fill='red')
       + gg.geom_point(gg.aes(y='y'), data=df, size=3))
  return p

############################################################
# Specialized plots for 2D problems
BLUE = '#084594'
RED = '#e41a1c'


def gen_2d_grid(plot_range: float) -> np.ndarray:
  """Generates a 2D grid for data in a certain_range."""
  data = []
  x_range = np.linspace(-plot_range, plot_range)
  for x1 in x_range:
    for x2 in x_range:
      data.append((x1, x2))
  return np.vstack(data)


def _gen_samples_2d(enn_sampler: testbed_base.EpistemicSampler,
                    x: chex.Array,
                    num_samples: int,
                    categorical: bool = False) -> pd.DataFrame:
  """Generate posterior samples at x (not implemented for all posterior)."""
  # Generate the samples
  data = []
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed=0))
  for seed in range(num_samples):
    net_out = enn_sampler(x, next(rng))
    y = jax.nn.softmax(net_out)[:, 1] if categorical else net_out[:, 0]
    df = pd.DataFrame({'x0': x[:, 0], 'x1': x[:, 1], 'y': y, 'seed': seed})
    data.append(df)
  return pd.concat(data)


def _agg_samples_2d(sample_df: pd.DataFrame) -> pd.DataFrame:
  """Aggregate ENN samples for plotting."""
  def pct_95(x):
    return np.percentile(x, 95)
  def pct_5(x):
    return np.percentile(x, 5)
  enn_df = (sample_df.groupby(['x0', 'x1'])['y']
            .agg([np.mean, np.std, pct_5, pct_95]).reset_index())
  enn_df = enn_df.rename({'mean': 'y'}, axis=1)
  enn_df['method'] = 'enn'
  return enn_df


def _gen_problem_2d(
    problem: classification_envlikelihood.ClassificationEnvLikelihood,
    x: chex.Array,
) -> pd.DataFrame:
  """Generate underlying problem dataset."""
  assert x.shape[1] == 2
  logits = problem._logit_fn(x)  # pylint:disable=protected-access
  test_probs = jax.nn.softmax(logits)[:, 1]
  np_data = np.hstack([x, test_probs[:, None]])
  problem_df = pd.DataFrame(np_data, columns=['x0', 'x1', 'y'])
  problem_df['method'] = 'true_function'
  return problem_df


def _make_train_2d(
    problem: classification_envlikelihood.ClassificationEnvLikelihood):
  data = problem.train_data
  return pd.DataFrame(np.hstack([data.x, data.y]), columns=['x0', 'x1', 'y'])


def _plot_default_2d(problem_df: pd.DataFrame,
                     enn_df: pd.DataFrame,
                     train_df: pd.DataFrame) -> gg.ggplot:
  """Side-by-side plot comparing ENN and true function."""
  p = (gg.ggplot(pd.concat([problem_df, enn_df]))
       + gg.aes(x='x0', y='x1', fill='y')
       + gg.geom_tile()
       + gg.geom_point(data=train_df, size=3, stroke=1.5, alpha=0.7)
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0.5)
       + gg.facet_wrap('method')
       + gg.theme(figure_size=(12, 5))
       + gg.ggtitle('Comparing ENN and true probabilities')
      )
  return p


def _plot_expanded_2d(problem_df: pd.DataFrame,
                      enn_df: pd.DataFrame,
                      train_df: pd.DataFrame) -> gg.ggplot:
  """Side-by-side plot comparing ENN and true function with pct_5, pct_95."""
  plt_df = pd.melt(enn_df, id_vars=['x0', 'x1'],
                   value_vars=['y', 'pct_5', 'pct_95'])
  plt_df['variable'] = plt_df.variable.apply(lambda x: 'enn:' + x)
  problem_df['value'] = problem_df['y']
  problem_df['variable'] = 'true_function'
  p = (gg.ggplot(pd.concat([problem_df, plt_df]))
       + gg.aes(x='x0', y='x1', fill='value')
       + gg.geom_tile()
       + gg.geom_point(gg.aes(fill='y'), data=train_df, size=3, stroke=1.5,
                       alpha=0.7)
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0.5)
       + gg.facet_wrap('variable')
       + gg.theme(figure_size=(12, 10))
       + gg.ggtitle('Comparing ENN and true probabilities'))
  return p


def _plot_error_2d(problem_df: pd.DataFrame, enn_df: pd.DataFrame,
                   train_df: pd.DataFrame) -> gg.ggplot:
  """Single plot of error in ENN."""
  plt_df = pd.merge(
      enn_df, problem_df, on=['x0', 'x1'], suffixes=('_enn', '_problem'))
  p = (gg.ggplot(plt_df)
       + gg.aes(x='x0', y='x1', fill='y_problem - y_enn')
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0)
       + gg.geom_tile()
       + gg.geom_point(gg.aes(x='x0', y='x1', fill='y'), data=train_df, size=3,
                       stroke=1.5, inherit_aes=False, show_legend=False)
       + gg.theme(figure_size=(7, 5))
       + gg.ggtitle('Error in ENN mean estimation')
      )
  return p


def _plot_std_2d(enn_df: pd.DataFrame,
                 train_df: pd.DataFrame) -> gg.ggplot:
  """Single plot of standard deviation in ENN predications."""
  p = (gg.ggplot(enn_df)
       + gg.aes(x='x0', y='x1', fill='std')
       + gg.scale_fill_gradient2('white', '#005a32', '#ffff33', midpoint=0.1)
       + gg.geom_tile()
       + gg.geom_point(gg.aes(x='x0', y='x1', colour='y'), data=train_df,
                       size=3, inherit_aes=False, show_legend=False, alpha=0.7)
       + gg.scale_colour_gradient(BLUE, RED, limits=[0, 1])
       + gg.theme(figure_size=(7, 5))
       + gg.ggtitle('Standard deviation in ENN predications')
       )
  return p


def _plot_enn_samples_2d(sample_df: pd.DataFrame,
                         train_df: pd.DataFrame) -> gg.ggplot:
  """Plot realizations of enn samples."""
  p = (gg.ggplot(sample_df)
       + gg.aes(x='x0', y='x1', fill='y')
       + gg.geom_tile()
       + gg.geom_point(data=train_df, size=3, stroke=1.5)
       + gg.scale_fill_gradient2(BLUE, 'white', RED, midpoint=0.5)
       + gg.facet_wrap('seed', labeller='label_both')
       + gg.theme(figure_size=(18, 12), panel_spacing=0.1)
       + gg.ggtitle('ENN sample realizations')
      )
  return p


def generate_2d_plots(
    true_model: classification_envlikelihood.ClassificationEnvLikelihood,
    enn_sampler: testbed_base.EpistemicSampler,
    num_samples: int = 20) -> Dict[str, gg.ggplot]:
  """Generates a sequence of plots for debugging."""
  x = gen_2d_grid(3)
  sample_df = _gen_samples_2d(enn_sampler, x, num_samples, categorical=True)
  enn_df = _agg_samples_2d(sample_df)
  problem_df = _gen_problem_2d(true_model, x)
  train_df = _make_train_2d(true_model)
  return {
      'enn': _plot_default_2d(problem_df, enn_df, train_df),
      'more_enn': _plot_expanded_2d(problem_df, enn_df, train_df),
      'err_enn': _plot_error_2d(problem_df, enn_df, train_df),
      'std_enn': _plot_std_2d(enn_df, train_df),
      'sample_enn': _plot_enn_samples_2d(sample_df, train_df),
  }
