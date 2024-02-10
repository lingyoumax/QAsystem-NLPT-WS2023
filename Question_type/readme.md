7.2.2024
So far, some data sets containing question type labels have been generated through GPT and other methods, and the number is about 1,500. The next step is to use these datasets to train a model that detects question types.

For the specific process, please refer to analysis.ipynb

9.2.2024
So far, for the problem of class imbalance, the proportion of negative samples and positive samples is used as the weight to calculate the weight of each column.
And some of the previously required measures to delete the data set can be discarded. In the training process, regularization and dynamic adjustment of learning rate and early stop are added to increase generalization ability.

10.2.2024  
Up to now, it has been optimized for multi-category problems with very low accuracy that occur in each training.
1. Use focal loss and weighted BCEloss as the total loss function to improve the category imbalance problem.
2. In terms of training strategy, first train based on the total data set, and then train based on all multi-category problem data sets. It is best to train based on half of the multi-category and half of the single category.
3. Translate into other languages and then back to English to achieve data enhancement.
<img width="677" alt="7b93c244d9d958152ac8ae4f6433e2f" src="https://github.com/lingyoumax/QAsystem-NLPT-WS2023/assets/43053906/cda93b54-0fa6-4af3-80e5-f1278f85305b">
This is the current indicator

