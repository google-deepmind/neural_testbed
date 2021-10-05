# Agent evaluation on real datasets

This package includes tools for loading real datasets from tf.dataset and
evaluating a TestbedAgent on the data by partitioning the data into train and
test splits and estimating the cross entropy loss on the test split after
training the agent on the training split.
