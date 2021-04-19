# The Neural Testbed

## Introduction
Neural testbed is a framework for assessing the quality of
uncertainty estimates in neural networks. The comparison is with a posterior
approximation computed from reference Gaussian process prior. The Gaussian
process prior is equivalent to a randomly initialized infinite-width neural
network.

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
posterior against a *known* reference posterior.

We use Gaussian processes as the generative model in our testbed building on the
connection between Gaussian processes and [infinite width neural networks]. In
doing so, we compare the uncertainty estimates against those that would be
output by an idealized infinite-width neural network.

For more information see our [paper]

## How do I get started?

The Neural Testbed has a very simple interface, specified in
[base.py](https://github.com/deepmind/neural_testbed/base.py):

```python
class EpistemicSampler(typing_extensions.Protocol):
  """Interface for drawing posterior samples from distribution.

  We are considering a model of data: y_i = f(x_i) + e_i.
  In this case the sampler should only model f(x), not aleatoric y.
  """

  def __call__(self, x: base.Array, seed: int = 0) -> base.Array:
    """Generate a random sample for epistemic f(x)."""


class TestbedAgent(typing_extensions.Protocol):
  """An interface for specifying a testbed agent."""

  def __call__(self,
               data: Data,  # (X: base.Array, Y: base.Array)
               prior: Optional[PriorKnowledge] = None) -> EpistemicSampler:
    """Sets up a training procedure given ENN prior knowledge."""
```

In order to submit to the testbed you need to define your own `TestbedAgent`: a
function taking training data and optional prior knowledge and outputting an
`EpistemicSampler`. The `EpistemicSampler` outputs an approximate posterior
sample for the unknown function `f` at a given `x`, for a given `seed`. Given a
`TestbedAgent` the `neural_testbed` will handle everything else and output a
performance measure for the quality of the uncertainty estimation of the agent.



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
    title={Epistemic Neural Networks},
    author={},
    booktitle={Neural Information Processing Systems},
    year={2021},
    url={https://arxiv.org/}
}
```



[paper]:https://arxiv.org/
[leaderboard]: https://colab.research.google.com/github/deepmind/neural_testbed/blob/master/leaderboard/neural_testbed.ipynb
[infinite width neural networks]:https://arxiv.org/abs/1806.07572
