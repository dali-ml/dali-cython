import sys
sys.path.append("..")
import unittest

from dali import Tensor

class TensorTests(unittest.TestCase):
    def test_swapaxes(self):
        t = Tensor.zeros((1, 2, 3, 4))
        t = t.swapaxes(0, -1)
        self.assertEqual((4, 2, 3, 1), t.shape)
