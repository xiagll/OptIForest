import os
import sys
import time

# temporary solution for relative imports in case iffad is not installed
# if iffad is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '../..')))

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import IsolationForest

from iffad.detectors import LSHiForest


""""""
num_ensemblers = 100
threshold = 55
branch = 0

glass_df = pd.read_csv('data/glass.csv', header=None)  # pandas.DataFrame is returned by pandas.read_csv()
X = glass_df.values[:, :-1]  # numpy.ndarray is returned by pandas.DataFrame.values()
ground_truth = glass_df.values[:, -1]

detectors = [("sklearn.ISO", IsolationForest(n_estimators=num_ensemblers)),
             ("L2SH", LSHiForest('L2SH', num_ensemblers, threshold, branch))]  # L2HiForest for L2 distance with expected branch factor e

for i, (dtc_name, dtc) in enumerate(detectors):
    print("\n" + dtc_name + ":")
    AUC = []
    PR_AUC = []
    Traintime = []
    Testtime = []
    for j in range(10):
        start_time = time.time()

        dtc.fit(X)

        train_time = time.time() - start_time

        y_pred = dtc.decision_function(X)

        test_time = time.time() - start_time - train_time

        auc1 = roc_auc_score(ground_truth, -1.0*y_pred)
        AUC.append(auc1)
        pr_auc = average_precision_score(ground_truth, -1.0*y_pred)
        PR_AUC.append(pr_auc)

        Traintime.append(train_time)
        Testtime.append(test_time)
    mean_auc = np.mean(AUC)
    std_auc = np.std(AUC)
    mean_pr = np.mean(PR_AUC)
    std_pr = np.std(PR_AUC)
    mean_traintime = np.mean(Traintime)
    mean_testtime = np.mean(Testtime)

    print("\tAUC score:\t", mean_auc)
    print("\tAUC std:\t", std_auc)
    print("\tPR score:\t", mean_pr)
    print("\tPR std:\t", std_pr)
    print("\tTraining time:\t", mean_traintime)
    print("\tTesting time:\t", mean_testtime)
