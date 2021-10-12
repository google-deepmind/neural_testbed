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
"""Display functions for leaderboard entries."""

from typing import Sequence

from neural_testbed.leaderboard import score
import numpy as np
import pandas as pd


SCORE_COL = 'kl_estimate'
DISPLAY_COLS = (
    'agent_name', 'normalized_kl', 'normalized_stderr',
    'mean_test_acc', 'mean_train_seconds', 'mean_evaluation_seconds'
)


def _stderr(x):
  return np.std(x) / np.sqrt(len(x))


def _extract_tau_data(data: score.LeaderboardData, tau: int) -> pd.DataFrame:
  assert tau in data.df.tau.unique()
  return data.df[data.df.tau == tau].copy()


def _compute_mean(df: pd.DataFrame, column_name: str):
  """Computes mean running time based on column column_name."""
  if column_name not in df.columns:
    df[column_name] = 0
  mean_df = (df.groupby('agent_name')[column_name]
             .agg([np.mean])
             .rename({'mean': 'mean_' + column_name}, axis=1)
             .reset_index())

  return mean_df


def _compute_stderr(df: pd.DataFrame, num_seed_per_class: int = 10):
  """Computes stderr by grouping the problems based on their seeds."""
  assert 'seed' in df.columns
  df['seed_class'] = df['seed'].apply(lambda x: x % num_seed_per_class)
  kl_seed_df = df.groupby(['agent_name',
                           'seed_class'])['kl_estimate'].mean().reset_index()
  stderr_df = kl_seed_df.groupby(['agent_name'
                                 ])['kl_estimate'].agg([_stderr]).reset_index()
  stderr_df = stderr_df.rename({'_stderr': 'stderr_kl'}, axis=1)

  return stderr_df


def compute_normalization(data: score.LeaderboardData,
                          agent_name: str = 'baseline',
                          tau: int = 1) -> float:
  df = _extract_tau_data(data, tau)
  return df[df.agent_name == agent_name]['kl_estimate'].mean()


def compute_ranking(data: score.LeaderboardData,
                    num_seed_per_class: int = 10,
                    tau: int = 1,
                    kl_limit: float = 1e6) -> pd.DataFrame:
  """Compute the ranking based on the average KL divergence."""
  # Subsample data to a specific tau
  df = _extract_tau_data(data, tau)

  if 'baseline:uniform_class_probs' in data.df.agent_name.unique():
    normalizing_score = compute_normalization(
        data, 'baseline:uniform_class_probs', tau)
  else:
    print('WARNING: uniform_class_probs agent not included in data, '
          'no normalization is applied.')
    normalizing_score = 1

  # Calculate the mean KL
  rank_df = _compute_mean(df, column_name=SCORE_COL)
  # Calculate the std error
  stderr_df = _compute_stderr(df, num_seed_per_class=num_seed_per_class)
  rank_df = pd.merge(rank_df, stderr_df, on='agent_name', how='left')
  # Calculate the mean test acc
  testacc_df = _compute_mean(df, column_name='test_acc')
  rank_df = pd.merge(rank_df, testacc_df, on='agent_name', how='left')
  # Calculate the mean training time
  traintime_df = _compute_mean(df, column_name='train_seconds')
  rank_df = pd.merge(rank_df, traintime_df, on='agent_name', how='left')
  # Calculate the mean evaluation time
  evaltime_df = _compute_mean(df, column_name='evaluation_seconds')
  rank_df = pd.merge(rank_df, evaltime_df, on='agent_name', how='left')
  # TODO(author2): Work out what's going wrong with unhashable hypers e.g. list.
  for var in data.sweep_vars:
    try:
      df[var].unique()
    except TypeError:
      df[var] = df[var].astype(str)
  hyper_df = df[data.sweep_vars].drop_duplicates()
  df = pd.merge(rank_df, hyper_df, on='agent_name', how='left')
  # Adding in the normalized values
  df['normalized_kl'] = df['mean_kl_estimate'] / normalizing_score
  df['normalized_stderr'] = df['stderr_kl'] / normalizing_score
  out_df = df.sort_values('normalized_kl').reset_index().drop({'index'}, axis=1)
  # TODO(author2): Find a better way to limit KL in output plots.
  return out_df[out_df.normalized_kl < kl_limit]


def display_ranking_df(
    data: score.LeaderboardData,
    display_cols: Sequence[str] = DISPLAY_COLS,
    num_seed_per_class: int = 10,
    tau: int = 1) -> pd.DataFrame:
  """Display the ranking based on the average KL divergence."""
  display_cols = list(display_cols)
  score_df = compute_ranking(data, num_seed_per_class, tau)
  return score_df[display_cols]
