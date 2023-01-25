import unittest

import pandas as pd

from detectors.opt_forest import *
from detectors.sampling  import *
from scipy.spatial import distance
from detectors.opt import E2LSH
from detectors.opt import AngleLSH

class TestOptForest(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('../data/ad.csv', header=None)
        self._X = data.values[:, :-1]

    def test_decision_function(self):
        num_of_tree = 100
        Opt_forest = OptForest(num_of_tree, VSSampling(num_of_tree), E2LSH(norm=2), 403, 0, distance.euclidean)
        Opt_forest.fit(self._X)
        # l2hash_forest.display()

        x = self._X[0:10, :]
        score = Opt_forest.decision_function(x)
        print("score: ", score)
        print("average branch factor: ", Opt_forest.get_avg_branch_factor())
        self.assertTrue(Opt_forest is not None)