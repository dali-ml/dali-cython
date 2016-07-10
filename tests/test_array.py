import sys
sys.path.append("..")
import unittest

from dali import Array

class ArrayTests(unittest.TestCase):
    def test_swapaxes(self):
        ar = Array.empty((1, 2, 3, 4))
        ar = ar.swapaxes(0, -1)
        self.assertEqual((4, 2, 3, 1), ar.shape)
