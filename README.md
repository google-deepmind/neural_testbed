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
    git clone https://github.com/deepmind/neural_testbed.git
    cd neural_testbed
    pip install .
    ```

3. **Optional**: run the tests by executing `./test.sh` from the `neural_testbed` main directory.




## Baseline agents

In addition to our testbed code, we release a collection of benchmark agents.
These include the full sets of hyperparameter sweeps necessary to reproduce the paper's results, and can serve as a great starting point for new research.
You can have a look at these implementations in the [`agents/factories/`](neural_testbed/agents/factories) folder.

We recommended you get started with our [colab tutorial].
After [intallation](#installation) you can also run an agent directly by executing the following command from the main directory of `neural_testbed`:

```bash
python -m neural_testbed.experiments.run --agent_name=ensemble_plus
```

By default, this will save the results for that agent to csv at `/tmp/neural_testbed`.
You can control these options by flags in the run file.
In particular, you can run the agent on the whole sweep of tasks in the Neural Testbed by specifying the flag `--problem_id=SWEEP`.


## Citing

If you use `neural_testbed` in your work, please cite the accompanying [paper]:

```bibtex
@misc{osband2021evaluating,
      title={Evaluating Predictive Distributions: Does Bayesian Deep Learning Work?},
      author={Ian Osband and Zheng Wen and Seyed Mohammad Asghari and Vikranth Dwaracherla and Botao Hao and Morteza Ibrahimi and Dieterich Lawson and Xiuyuan Lu and Brendan O'Donoghue and Benjamin Van Roy},
      year={2021},
      eprint={2110.04629},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



[paper]:https://arxiv.org/abs/2110.04629
[colab tutorial]: https://colab.research.google.com/github/deepmind/neural_testbed/blob/master/neural_testbed/tutorial.ipynb

