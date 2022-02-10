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

"""Loading a realdata problem for the testbed."""
from absl import logging
import haiku as hk
from neural_testbed import base as testbed_base
from neural_testbed import likelihood
from neural_testbed.real_data import data_sampler
from neural_testbed.real_data import datasets
from neural_testbed.real_data import sweep
from neural_testbed.real_data import utils


def problem_from_id(problem_id: str) -> testbed_base.TestbedProblem:
  """Factory method to load realdata problem from problem_id.

  This is a user facing function and its only job is to translate problem_id
  to  prior kowledge.
  Args:
    problem_id: a string representing a standard problem in the leaderboard.
  Returns:
    A testbed problem.
  """

  logging.info('Loading problem_id: %s', problem_id)

  try:
    problem_config = sweep.SETTINGS[problem_id]
  except ValueError as value_error:
    raise ValueError(f'Unrecognised problem_id={problem_id}') from value_error

  return problem_from_config(problem_config)


def problem_from_config(
    problem_config: sweep.ProblemConfig) -> testbed_base.TestbedProblem:
  """Returns a testbed problem given a problem config."""
  assert problem_config.prior_knowledge.num_classes > 0

  if problem_config.prior_knowledge.num_classes > 1:
    return _load_classification(problem_config)
  else:
    return _load_regression(problem_config)


def _load_classification(
    problem_config: sweep.ProblemConfig) -> testbed_base.TestbedProblem:
  """Load a classification problem from problem_config."""
  rng = hk.PRNGSequence(problem_config.seed)
  prior_knowledge = problem_config.prior_knowledge

  dataset_info = datasets.DATASETS_SETTINGS[problem_config.dataset_name]
  # Update num_train of the dataset
  dataset_info.num_train = problem_config.prior_knowledge.num_train

  train_data = utils.load_classification_dataset(
      dataset_info=dataset_info, split='train',)
  test_data = utils.load_classification_dataset(
      dataset_info=dataset_info, split='test',)

  realdata_sampler = data_sampler.RealDataSampler(
      train_data=train_data,
      test_sampler=data_sampler.make_local_sampler(test_data),
      tau=prior_knowledge.tau,
  )

  sample_based_kl = likelihood.CategoricalKLSampledXSampledY(
      num_test_seeds=problem_config.num_test_seeds,
      num_enn_samples=problem_config.num_enn_samples,
      key=next(rng),
      num_classes=prior_knowledge.num_classes,
  )

  sample_based_kl = likelihood.add_classification_accuracy_ece(
      sample_based_kl,
      num_test_seeds=int(1_000 / prior_knowledge.tau) + 1,
      num_enn_samples=100,
      num_classes=prior_knowledge.num_classes,
  )

  return likelihood.SampleBasedTestbed(
      data_sampler=realdata_sampler,
      sample_based_kl=sample_based_kl,
      prior_knowledge=prior_knowledge,
  )


def _load_regression(
    problem_config: sweep.ProblemConfig) -> testbed_base.TestbedProblem:
  """Load a regression problem from problem_config."""
  rng = hk.PRNGSequence(problem_config.seed)
  prior_knowledge = problem_config.prior_knowledge

  train_data, test_data = utils.load_regression_dataset(
      dataset_name=problem_config.dataset_name)

  realdata_sampler = data_sampler.RealDataSampler(
      train_data=train_data,
      test_sampler=data_sampler.make_global_sampler(test_data),
      tau=prior_knowledge.tau,
  )

  sample_based_kl = likelihood.GaussianSampleKL(
      # This KL estimator cannot handle very large num_test_seed * tau
      num_test_seeds=int(problem_config.num_test_seeds
                         / prior_knowledge.tau) + 1,
      num_enn_samples=problem_config.num_enn_samples,
      enn_sigma=prior_knowledge.noise_std,
      key=next(rng),
  )
  return likelihood.SampleBasedTestbed(
      data_sampler=realdata_sampler,
      sample_based_kl=sample_based_kl,
      prior_knowledge=prior_knowledge,
  )
