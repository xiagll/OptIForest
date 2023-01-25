import unittest

import pandas as pd
import numpy as np

from detectors.opt import *


class TestE2LSH(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('../data/ad.csv', header=None)
        self._X = data.values[:, :-1]
        indices = range(len(self._X))
        # Uncomment the following code for continuous values
        self._data = np.c_[indices, self._X]

        self._elsh = E2LSH()

    def test_get_lsh_type(self):
        self.assertEqual(self._elsh.get_lsh_type(), "L2LSH")

    def test_fit(self):
        self._elsh.fit(self._data)
        self._elsh.display_hash_func_parameters()
        self.assertTrue(self._elsh.A_array is not None)

    def test_get_hash_value(self):
        self._elsh.fit(self._data)
        self._elsh.display_hash_func_parameters()
        key_stat = {}
        for x in self._data:
            key = self._elsh.get_hash_value(x[1:],1)
            if key not in key_stat:
                key_stat[key] = 1
            else:
                key_stat[key] += 1
        print(key_stat)
        self.assertTrue(len(key_stat) >= 2)

class TestAngleLSH(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('../data/ad.csv', header=None)
        self._X = data.values[:, :-1]
        indices = range(len(self._X))
        # Uncomment the following code for continuous values
        self._data = np.c_[indices, self._X]

        self._alsh = AngleLSH()

    def test_get_lsh_type(self):
        self.assertEqual(self._alsh.get_lsh_type(), "AngleLSH")

    def test_fit(self):
        self._alsh.fit(self._data)
        self._alsh.display_hash_func_parameters()
        self.assertTrue(self._alsh._weights is not None)

    def test_get_hash_value(self):
        self._alsh.fit(self._data)
        self._alsh.display_hash_func_parameters()
        key_stat = {}
        for x in self._data:
            key = self._alsh.get_hash_value(x[1:],1)
            if key not in key_stat:
                key_stat[key] = 1
            else:
                key_stat[key] += 1
        print(key_stat)
        self.assertTrue(len(key_stat) >= 2)

class TestHierHash(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv('../data/ad.csv', header=None)
        self._X = data.values[:, :-1]
        indices = range(len(self._X))
        # Uncomment the following code for continuous values
        self._data = np.c_[indices, self._X]

        self._hlsh = HierHash()

    def test_get_hash_type(self):
        self.assertEqual(self._hlsh.get_hash_type(), "Hier_Hash")

    def test_fit(self):
        self._hlsh.fit([self._data[0],self._data[1]],[1,-1],[1000,2279])
        self.assertTrue(self._hlsh.centers is not None)

    def test_get_hash_value(self):
        self._hlsh.fit([self._data[0],self._data[1]],[1,-1],[1000,2279])
        key_stat = {}
        for x in self._data:
            key = self._hlsh.get_hash_value(x[1:])
            if key not in key_stat:
                key_stat[key] = 1
            else:
                key_stat[key] += 1
        print(key_stat)
        self.assertTrue(len(key_stat) >= 2)