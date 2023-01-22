# Optimal Isolation Forest for Anomaly Detection
This repository contains the code for the experiments of the paper "OptIForest: Optimal Isolation Forest for Anomaly Detection".

# Requirement
- numpy==1.20.1
- sklearn==0.22.1
- pandas==1.4.1

# Dataset
We evaluate all methods on 20 widely-used benchmark datasets, which are available in [public UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) and [Kaggle Repository](https://www.kaggle.com/datasets), while other datasets, like ``AD``, ``Celeba``, ``Census`` and ``Cover``, can be found in public [ADRepository](https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets).

# Experiment
### You can try different `--alpha` (non-IID data distribution), `--local_ep` (number of local epochs) to see how the attack performance changes. 
### An example for running the code:
    import pandas as pd  
