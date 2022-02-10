# Agent factories

This folder contains the implementation of benchmark agents and utilities for
developing new `TestbedAgent`s. The benchmark agents can be used as starting
point for developing new agents. This folder includes a diverse set of agents
including classic point estimators like KNN and random forest to different
neural network based agents like Bayes by backprop and hypermodel.

- For each agent (e.g., agent_name=ensemble), the implementation can be found
  in `{agent_name}.py`.

-   `sweeps`: contains the agent sweeps tuned for
  -   `testbed` which is the neural testbed with input dimension of 2, 10, 100.
  -   `testbed_2d` which is neural testbed only with input dimension of 2.
  -   `real_data` which is a collection of real datasets.
