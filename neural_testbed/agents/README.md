# Agents

## Summary

This package contains the implementation of benchmark agents and utilities for
developing new `TestbedAgent`s. The benchmark agents in the factories subpackage
can be used as starting point for developing new agents. This subpackage
includes a diverse set of agents including classic point estimators like KNN and
random forest to different neural network based agents like Bayes by backprop
and hypermodel.

## Evaluating new agents on the testbed

In order to get familiar with the interface a good starting point is the
`baseline:uniform_class_probs` in the
[`baselines.py`](neural_testbed/agents/factories/baselines.py) module. This is
the simplest possible agent that ignores the input and always outputs equal
logits for all classes.

To see how an agent can use the training data look at `average class probability
agent` in the [`baselines.py`](neural_testbed/agents/factories/baselines.py)
module. This agent uses the training data to calculate the frequency of each
class in the training data and output the relative class frequency for each
class irrespective of the input at test time.

A fully fledged agent is the `ensemble` agent implemented in
[`ensemble.py`](neural_testbed/agents/factories/ensemble.py) module. This agent
trains an ensemble of MLP models and samples one based on the index received at
test time.

## Performance of the benchmark agents

The table below summarizes the main agents that are included and the references
for the canonical implementation. It also lists the main hyperparameters we
experimented with for these agents.

| **agent**  | **description**              | **hyperparameters**             |
| ---------- | ---------------------------- | ------------------------------- |
| mlp        | Vanilla MLP                  | weight decay                    |
| ensemble   | [Deep Ensemble]              | weight decay, ensemble size     |
| dropout    | [Dropout]                    | weight decay, network, dropout  |
:            :                              : rate                            :
| bbb        | [Bayes by Backprop]          | prior mixture, network, early   |
:            :                              : stopping                        :
| sgmcmc     | [Stochastic Langevin MCMC]   | learning rate, prior, momentum  |
| ensemble+  | [Ensemble + prior functions] | weight decay, ensemble size,    |
:            :                              : prior scale, bootstrap          :
| hypermodel | [Hypermodel]                 | weight decay, prior, bootstrap, |
:            :                              : index dimension                 :

The figure below shows the score of each of the above agents on the testbed when
measured for the best hyperparameter choice.

![Benchmark agent performance on the testbed](statics/images/benchmark_agents_kl_estimates.png)

[Deep Ensemble]:https://arxiv.org/abs/1612.01474
[Dropout]:https://arxiv.org/abs/1506.02142
[Bayes by Backprop]:https://arxiv.org/abs/1505.05424
[Stochastic Langevin MCMC]:https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
[Ensemble + prior functions]:https://arxiv.org/abs/1806.03335
[Hypermodel]:https://arxiv.org/abs/2006.07464
