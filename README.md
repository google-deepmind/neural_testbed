# The Neural Testbed

![Neural Testbed Logo](statics/images/neural_testbed_logo.png)


## Introduction

Posterior predictive distributions quantify uncertainties ignored by point estimates.
The `neural_testbed` provides tools for the systematic evaluation of agents that generate such predictions.
Crucially, these tools assess not only the quality of marginal predictions per input, but also joint predictions given many inputs.
Joint distributions are often critical for useful uncertainty quantification, but they have been largely overlooked by the Bayesian deep learning community.

This library automates the evaluation and analysis of learning agents:

- Synthetic neural-network-based generative model.
- Evaluate predictions beyond marginal distributions.
- Reference implementations of benchmark agents (with tuning).

For a more comprehensive overview, see the accompanying [paper](paper link).


## Technical overview

We outline the key high-level interfaces for our code in [base.py](neural_testbed/base.py):

- `EpistemicSampler`: Generates a random sample from agent's predictive distribution.
- `TestbedAgent`: Given data, prior_knowledge outputs an EpistemicSampler.
- `TestbedProblem`: Reveals training_data, prior_knowledge. Can evaluate the quality of an EpistemicSampler.

If you want to evaluate your algorithm on the testbed, you simply need to define your `TestbedAgent` and then run it on our [experiment.py](neural_testbed/experiment/experiment.py)

```python
def run(agent: testbed_base.TestbedAgent,
        problem: testbed_base.TestbedProblem) -> testbed_base.ENNQuality:
  """Run an agent on a given testbed problem."""
  enn_sampler = agent(problem.train_data, problem.prior_knowledge)
  return problem.evaluate_quality(enn_sampler)
```

The `neural_testbed` takes care of the evaluation/logging within the `TestbedProblem`.
This means that the experiment will automatically output data in the correct format.
This makes it easy to compare results from different codebases/frameworks, so you can focus on agent design.


## How do I get started?

If you are new to `neural_testbed` you can get started in our [colab tutorial].
This Jupyter notebook is hosted with a free cloud server, so you can start coding right away without installing anything on your machine.
After this, you can follow the instructions below to get `neural_testbed` running on your local machine:


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




## Baseline agents

In addition to our testbed code, we release a collection of benchmark agents.
These include the full sets of hyperparameter sweeps necessary to reproduce the paper's results, and can serve as a great starting point for new research.
You can have a look at these implementations in the [`agents/factories/`](neural_testbed/agents/factories) folder.

We recommended you get started with our [colab tutorial](link).
After [intallation](#installation) you can also run an agent directly:

```bash
python -m neural_testbed.experiments.enn.run --agent_name=ensemble_plus
```

By default, this will save the results for that agent to csv at `/tmp/neural_testbed`.
You can control these options by flags in the run file.


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

