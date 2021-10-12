# The Neural Testbed

![Neural Testbed Logo](statics/images/neural_testbed_logo.png)

## What is The Neural Testbeed
The Neural Testbed is a framework for assessing and comparing performance of
uncertainty estimators, i.e., models that output posterior predictive
distributions rather than a point estimate. We call such models an agent. The
Testbed implements synthetic data generating processes and streamlines the
process of sampling data, training agents, and evaluating test performance by
estimating KL-loss for marginal and *high-order joint* predictions.

The Neural Testbed consists of a carefully-selected collection of synthetic supervised learning
tasks, where we can compare the quality of a learned (usually neural network)
posterior against a reference posterior.
The generative model is based around a random 2-layer MLP. Since independent
data can be generated for each execution, the Testbed can drive insight and
multiple iterations of algorithm development without risk of overfitting to a
fixed dataset.


The goal is to provide a reliable and coherent evaluation framework that accelarate the
development of computational tools for approximate Bayesian inference that
leverage some of the strengths of deep learning.

This repository also includes the implementation of a
comprehensive set of benchmark agents and their performance on the testbed. In
addition, we provide utilities for evaluating the agents on real dataset where
the estimate of the cross-entropy loss on the test data, i.e.,
sample mean of the negative log-likelihoods is reported as the metric.


For more information see our [paper]

## How do I get started?

A great starting point is the [tutorial] colab. The colab is extensively documented and allows running the code without installing anything locally.
You can also run any of the included agents on a task on a local machine after [installing](#installation) the package:

```bash
python -m neural_testbed.experiments.enn.run
```
You can control the agent and the task using the flags in this file.

The agent implementations are in the [`agents/factories/`](neural_testbed/agents/factories) package. Each agent
implementation also includes a config definition that specifies the
hyperparameters of that agent and a set of *sweeps* that represents
hyperparameter sweeps used to optimize the performance of the agent.

## Evaluating a new agent

In order to evaluate a new agent on the testbed you need to define your own `TestbedAgent` that implementes the interface, specified in
[base.py](neural_testbed/base.py):

```python
class EpistemicSampler(typing_extensions.Protocol):
  """Interface for drawing posterior samples from distribution.

  For classification this should represent the class *logits*.
  For regression this is the posterior sample of the function f(x).
  Assumes a batched input x.
  """

  def __call__(self, x: chex.Array, key: chex.PRNGKey) -> chex.Array:
    """Generate a random sample from approximate posterior distribution."""


class TestbedAgent(typing_extensions.Protocol):
  """An interface for specifying a testbed agent."""

  def __call__(self, data: Data, prior: PriorKnowledge) -> EpistemicSampler:
    """Sets up a training procedure given ENN prior knowledge."""
```

A `TestbedAgent` is a function taking training data and optional prior knowledge
and outputting an
`EpistemicSampler`. The `EpistemicSampler` outputs an approximate posterior
sample for the unknown function `f` at a given `x`, for a given random `key`.
Given a `TestbedAgent` the `neural_testbed` will handle everything else and
output a performance measure for the quality of the uncertainty estimation of
the agent. It can also evaluate the agent on real dataset by estimating the test
cross-entropy loss.



### Installation

We have tested `neural_testbed` on Python 3.7. To install the dependencies:

1.  **Optional**: We recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies, so as not to clobber your system installation:

    ```bash
    python3 -m venv neural_testbed
    source neural_testbed/bin/activate
    pip install --upgrade pip setuptools
    ```

2.  Install `neural_testbed` directly from [github](https://github.com/deepmind/neural_testbed):

    ```bash
    pip install git+https://github.com/deepmind/neural_testbed
    ```

3. **Optional**: run the tests by executing `./test.sh` from the root directory.


## Citing

If you use `neural_testbed` in your work, please cite the accompanying [paper]:

```bibtex
@inproceedings{,
    title={Evaluating Predictive Distributions: Does Bayesian Deep Learning Work?},
    author={},
    booktitle={},
    year={2021},
    url={https://arxiv.org/}
}
```



[paper]:https://arxiv.org/
[tutorial]: https://colab.research.google.com/github/deepmind/neural_testbed/blob/master/neural_testbed/reports/tutorial.ipynb

