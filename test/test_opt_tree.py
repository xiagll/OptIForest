import unittest

import pandas as pd

from detectors.opt_tree import *
from detectors.opt import E2LSH


class TestL2HashTree(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('../data/ad.csv', header=None)
        self._X = data.values[:, :-1]
        self._lsh_family = E2LSH(norm=2)
        self._lsh_family.fit(self._X)


    def test_lsh_root_is_not_none(self):
        lsh_tree = LSHTree(self._lsh_family)
        lsh_tree.build(self._X)
        # lsh_tree.display()
        print("averge branch factor: ", lsh_tree.get_avg_branch_factor())
        self.assertTrue(lsh_tree._root is not None)

    def test_opt_root_is_not_none(self):
        opt_tree = HierTree(self._lsh_family, 403, 0)
        opt_tree.build(self._X)
        print("averge branch factor: ", opt_tree.get_avg_branch_factor())
        self.assertTrue(opt_tree._root is not None)

    def test_decision_function(self):
        opt_tree = HierTree(self._lsh_family, 403, 0)
        opt_tree.build(self._X)
        x = self._X[0]
        score = opt_tree.predict(1, x)
        print("score: ", score)
        self.assertTrue(score > 0)