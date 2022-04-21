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
"""This folder is designed to collect agent definitions for neural testbed.

Each file should contain:

-  **AgentConfig** = dataclass describing the config of the agent. This should
   be initialized with the chosen parameters as chosen by the experimental
   sweep on the testbed.
-  **AgentCtor**: AgentConfig -> TestbedAgent = constructor of the agent.
-  **AgentSweep**: None -> Sequence[AgentConfig] = the sequence of parameters
   necessary to run the sweep for the paper.

We might be able to express this structure through TypeVar.
"""

import dataclasses
from typing import Callable, Generic, Sequence, TypeVar

from neural_testbed import base as testbed_base

# TypeVar parameterizes in terms of this input
AgentConfig = TypeVar('AgentConfig')


@dataclasses.dataclass
class PaperAgent(Generic[AgentConfig]):
  default: AgentConfig
  ctor: Callable[[AgentConfig], testbed_base.TestbedAgent]
  sweep: Callable[[], Sequence[AgentConfig]]

