# Bandit experiments

This folder contains the code associated with a simple sequential decision
problem derived from the Neural Testbed. We consider a Thompson sampling agent
that selects actions randomly according to the probability it believes that
action is the optimal action.

-   `agents`: contains the code that converts a static testbed agent to a
    dynamic decision agent. Mostly, this involves annealing the amount of prior
    regularization as the agent observes more data.
-   `run`: contains an example of running the agents in these decision problems.
-   `thompson`: contains the core decision logic for a Thompson sampling agent.
