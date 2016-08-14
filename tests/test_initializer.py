import sys
import pickle
sys.path.append("..")
import unittest

from dali import Array, float32, float64, int32
import dali
import numpy as np

class InitializerTests(unittest.TestCase):
    def test_eye(self):
        ar = Array.empty((4, 4))
        dali.array.op.initializer.eye(2.0).assign(ar)
        self.assertTrue(np.allclose(np.eye(4) * 2.0, ar.get_value()))
