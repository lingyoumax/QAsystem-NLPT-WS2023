7.2.2024
So far, some data sets containing question type labels have been generated through GPT and other methods, and the number is about 1,500. The next step is to use these datasets to train a model that detects question types.

For the specific process, please refer to analysis.ipynb

9.2.2024
So far, for the problem of class imbalance, the proportion of negative samples and positive samples is used as the weight to calculate the weight of each column.
And some of the previously required measures to delete the data set can be discarded. In the training process, regularization and dynamic adjustment of learning rate and early stop are added to increase generalization ability.
