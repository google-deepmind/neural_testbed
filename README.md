# The Neural Testbed

## Introduction

The Neural Testbed is a framework for assessing and comparing performance of
uncertainty estimators (which we call agent).  The Testbed implements synthetic data generating
processes and streamlines the process of sampling data, training agents, and
evaluating test performance by estimating KL-loss for marginal and high-order
joint predictions.

### Uncertainty in deep learning

In many applications of machine learning and artificial intelligence, it is
required or beneficial to be able to reason under uncertainty:

-   **Bayesian theory sets the "gold standard" for probabilistic decision
    making**, but can quickly become intractable in large and complex systems
    with messy data.
-   **Deep learning has shown great success at scaling to large and complex
    datasets**, but typically focuses on point estimates. Where uncertainty
    estimates exist they are notoriously unreliable.

**Neural testbed is a framework for evaluating the quality of uncertainty
estimates in deep learning**. The goal is to provide a reliable and
coherent evaluation framework that accelarate the development of
computationaltools for approximate Bayesian inference that leverage some of the
strengths of deep learning.

You can see the current leaderboard at [leaderboard].

### What is the Neural Testbed?

The Neural Testbed is a carefully-selected collection of supervised learning
tasks, where we can compare the quality of a learned (usually neural network)
posterior against a reference posterior.

The generative model is based around a random 2-layer MLP. Since independent
data can be generated for each execution, the Testbed can drive insight and
multiple iterations of algorithm development without risk of overfitting to a
fixed dataset.  This repository also includes the implementation of a
comprehensive set of benchmark agents and their performance on the testbed. In
addition, we provide utilities for evaluating the agents on real dataset where
the estimate of the cross-entropy loss on the test data is taken to be the
sample mean of the negative log-likelihoods.

For more information see our [paper]

## How do I get started?

The Neural Testbed has a very simple interface, specified in
[base.py](https://github.com/deepmind/neural_testbed/base.py):

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

In order to submit to the testbed you need to define your own `TestbedAgent`: a
function taking training data and optional prior knowledge and outputting an
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
[leaderboard]: https://colab.research.google.com/github/deepmind/neural_testbed/blob/master/leaderboard/neural_testbed.ipynb
