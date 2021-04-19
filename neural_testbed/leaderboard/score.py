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

"""Scoring/validation capabilities for leaderboard entries."""

import copy
from typing import Any, Optional, Sequence, Tuple

import dataclasses
from neural_testbed import logging
from neural_testbed.leaderboard import sweep
import numpy as np
import pandas as pd

# Used to fill missing or non-numeric values
KL_FILL = 100
_AGENT_SUFFIX = '_agent'


@dataclasses.dataclass
class AgentData:
  """Contains cleaned data for a single ENN agent."""
  df: pd.DataFrame  # Data from full evaluation run
  name: str = 'agent'  # Name for the agent in plots etc
  score: float = KL_FILL  # Overall score on the testbed
  pct_health: float = 0.  # 1 for perfect data, 0 for all missing / NaN
  validated: bool = False  # Has this been validated
  xm_link: Optional[str] = None  # Link to experiment
  report_link: Optional[str] = None  # Link to experiment report


@dataclasses.dataclass
class LeaderboardData:
  """Contains cleaned data for a collections of agents."""
  agents: Sequence[AgentData]  # All of the consituent agents.
  sweep_vars: Optional[Sequence[str]] = None

  def __post_init__(self):
    """Form a cleaned version of the joined data."""
    self.df = _make_leaderboard_dataframe(self.agents, self.sweep_vars)
    self.names = [x.name for x in self.agents]


def join_metadata(df: pd.DataFrame) -> pd.DataFrame:
  """Joins data with GP settings based on gp_id."""
  assert 'gp_id' in df.columns
  metadata = copy.deepcopy(sweep.SETTINGS)

  data = []
  for gp_id, gp_kwargs in metadata.items():
    gp_params = {'gp_id': gp_id}
    gp_params.update(gp_kwargs)
    data.append(gp_params)
  gp_df = pd.DataFrame(data)

  # TODO(author2): Work out how to handle clash between agent sweep and gp_id
  # e.g. if an agent sweeps over temperature and so does the GP!
  return pd.merge(df, gp_df, on='gp_id', suffixes=(_AGENT_SUFFIX, ''))


def _clean_single_agent(df_in: pd.DataFrame,
                        leaderboard_sweep: Sequence[str],
                        agent_name: str = 'agent',
                        kl_fill: float = KL_FILL,
                        negative_tolerance: float = -1e-4,
                        verbose: bool = True) -> AgentData:
  """Validates and cleans the submission for a single agent."""
  df = df_in.copy()
  df = join_metadata(df)
  df['raw_kl_estimate'] = df['kl_estimate']
  gp_ids = df.gp_id.unique()

  # Giving the agent a name
  if 'agent_name' in df.columns:
    assert len(df.agent_name.unique()) == 1
    agent_name = df.agent_name.iloc[0]
  else:
    df['agent_name'] = agent_name
  if verbose:
    print('\n' + '+' * 80)
    print(f'Cleaning data for agent = {agent_name}')

  # Drop extra gp_id.
  extra_ids = [idx for idx in gp_ids if idx not in leaderboard_sweep]
  if extra_ids and verbose:
    print(f'WARNING: agent={agent_name} has {len(extra_ids)} extra gp_ids'
          f' these will be dropped:')
    print(extra_ids)
    df = df[~df.gp_id.isin(extra_ids)]

  # Fill missing gp_id
  missing_ids = [idx for idx in leaderboard_sweep if idx not in gp_ids]
  if missing_ids:
    fill_dict = {
        'agent_name': agent_name,
        'gp_id': missing_ids,
        'kl_estimate': kl_fill,
    }
    for col in df.columns:
      if len(df[col].unique()) == 1:
        fill_dict[col] = df[col].iloc[0]
    df = pd.concat([df, join_metadata(pd.DataFrame(fill_dict))])
    if verbose:
      print(f'WARNING: agent={agent_name} has {len(missing_ids)} missing gp_ids'
            f' (these will be filled with {kl_fill}')
      print(missing_ids)

  # Negative KL estimates
  negative_kl = df[df.kl_estimate < 0]
  num_negative = len(negative_kl)
  # You're only bad negative if even lower than negative tolerance
  bad_negative = len(df[df.kl_estimate < negative_tolerance])
  if num_negative:
    kl_values = negative_kl.kl_estimate
    df.loc[df.kl_estimate < 0, 'kl_estimate'] = 0
    if verbose:
      print(f'WARNING: agent={agent_name} has {num_negative} negative KL, '
            'these will be clipped at zero.')
      print(f'mean={kl_values.mean()}, min={kl_values.min()}')

  # Non-numeric KL estimates
  bad_kl = df[~np.isfinite(df.kl_estimate)]
  num_bad = len(bad_kl)
  if num_bad:
    df.loc[~np.isfinite(df.kl_estimate), 'kl_estimate'] = kl_fill
    if verbose:
      print(f'WARNING: agent={agent_name} has {num_bad} non-finite KL. '
            f'These values will be filled with {kl_fill}.\n')

  # Aggregate health of the entry
  total_bad = num_bad + bad_negative + len(extra_ids) + len(missing_ids)
  total_entries = len(df)
  df = df.assign(
      pct_finite=1 - num_bad/total_entries,
      pct_negative=1 - num_negative/total_entries,
      pct_extra=len(extra_ids)/total_entries,
      pct_missing=len(missing_ids)/total_entries,
      pct_health=1 - total_bad/total_entries,
  )
  data = AgentData(
      df=df,
      name=agent_name,
      score=df.kl_estimate.mean(),
      pct_health=1-total_bad/total_entries,
      validated=True,
  )
  return data


def _single_instance_or_list_to_list(var_instances: Any) -> Sequence[Any]:
  """Convert a potentially single-instance to a list of instance."""
  try:
    _ = len(var_instances)
    if isinstance(var_instances, str):
      var_instances = [var_instances]
  except TypeError:
    var_instances = [var_instances]
  return var_instances


def _make_variable_postfix(sub_vars: Any, sweep_vars: Sequence[str]) -> str:
  """Join hyperparameters to identify agent, e.g. num_ensemble=1_net=mlp."""
  sub_vars = _single_instance_or_list_to_list(sub_vars)
  assert len(sub_vars) == len(sweep_vars)
  return ','.join([f'{a}={b}' for a, b in zip(sweep_vars, sub_vars)])


def _maybe_add_links(agent: AgentData, entry: Any) -> AgentData:
  # Internal use only.
  return agent


def _load_single_entry(
    entry: Any,  # TODO(author2) turn this into a typevar for entries
    entry_loader: logging.EntryLoader,
    leaderboard_sweep: Sequence[str],
    verbose: bool = True,
) -> Tuple[Sequence[AgentData], Sequence[str]]:
  """Loads a single leaderboard entry and outputs list of AgentData."""
  df, sweep_vars = entry_loader(entry)

  if sweep_vars:
    # One entry for each of the sweep_vars
    data = []
    for sub_vars, sub_df in df.groupby(sweep_vars):
      post_fix = _make_variable_postfix(sub_vars, sweep_vars)
      agent = _clean_single_agent(
          sub_df, leaderboard_sweep, agent_name=f'{entry.name}:{post_fix}',
          verbose=verbose)
      data.append(_maybe_add_links(agent, entry))

  else:
    # The whole entry is just for one agent
    agent = _clean_single_agent(
        df, leaderboard_sweep, agent_name=entry.name, verbose=verbose)
    data = [_maybe_add_links(agent, entry)]

  return data, sweep_vars


def load_entries(
    leaderboard_entries: Any,  # TODO(author2): sort out this typing.
    entry_loader: logging.EntryLoader,
    leaderboard_sweep: Sequence[str] = sweep.CLASSIFICATION,
    verbose: bool = True,
) -> Tuple[Sequence[AgentData], Sequence[str]]:
  """Loads leaderboard entries and outputs a list of cleaned AgentData."""
  leaderboard_entries = _single_instance_or_list_to_list(leaderboard_entries)
  data = []
  sweep_vars = ['agent_name', 'notes', 'pct_health', 'report_link']
  for entry in leaderboard_entries:
    sub_data, sub_sweep = _load_single_entry(
        entry, entry_loader, leaderboard_sweep, verbose)
    data.extend(sub_data)
    sweep_vars.extend(sub_sweep)
  return data, sweep_vars


def combine_leaderboards(boards: Sequence[LeaderboardData]) -> LeaderboardData:
  """Combine multiple leaderboards into one."""
  agents = []
  sweep_vars = []
  names = []
  for board in boards:
    sweep_vars.extend(board.sweep_vars)
    for agent in board.agents:
      agents.append(agent)
      if agent.name not in names:
        names.append(names)
      else:
        raise ValueError(f'Duplicate agent={agent.name} encountered.'
                         ' You must rename agent to combine leaderboards.')
  return LeaderboardData(agents, list(set(sweep_vars)))  # For unique columns


def _make_leaderboard_dataframe(
    agents: Sequence[AgentData],
    sweep_vars: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
  """Process leaderboard entries into a unified dataframe."""
  data = []
  for agent in agents:
    data.append(agent.df.assign(report_link=agent.report_link))
  df = pd.concat(data)
  df['entry_name'] = df.agent_name.apply(lambda x: x.split(':')[0])
  df['task'] = df.gp_id.apply(lambda x: x.split('/')[0])
  if sweep_vars:
    for col in sweep_vars:
      try:
        df[col] = df[col].fillna('nan')  # Fixes bug in pandas groupby NaN.
      except KeyError:
        df[col] = 'nan'  # This column was completely missing
      except ValueError:
        pass  # This column did not want to be coerced to 'nan'
  return df
