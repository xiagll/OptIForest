import unittest

from detectors.opt_node import *


class TestOptNode(unittest.TestCase):
    def setUp(self):
        self._opt_nd = OptNode([], 10)

    def test_get_data_size(self):
        self._opt_nd.display()
        self.assertEqual(self._opt_nd.get_data_size(), 10)
