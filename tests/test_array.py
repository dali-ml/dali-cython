import sys
sys.path.append("..")
import unittest

from dali import Array, float32, float64, int32

import numpy as np

class ArrayTests(unittest.TestCase):
    def test_swapaxes(self):
        ar = Array.empty((1, 2, 3, 4))
        ar = ar.swapaxes(0, -1)
        self.assertEqual((4, 2, 3, 1), ar.shape)

    def test_cast(self):
    	ar = Array((1.1, 2.1, 3.1, 4.1), dtype=float32)
    	arint = ar.astype(int32).eval()
    	ardouble = ar.astype(float32).eval()

    	for i in range(ar.size):
    		self.assertEqual(float(ar[i]), float(ardouble[i]))
    		self.assertEqual(int(ar[i]), int(arint[i]))

    def test_tonumpy(self):
        expected = 0.9999

        arfloat64 = Array((0.5, 0.5, 0.9999,), dtype=float64)
        val = arfloat64[2].tonumpy()
        self.assertTrue(np.allclose(expected, val))

        arint32 = arfloat64.astype(int32).eval()
        val = arint32[2].tonumpy()
        self.assertTrue(np.allclose(round(expected), val))

        arfloat32 = Array((0.5, 0.5, 0.9999,), dtype=float32)
        val = arfloat32[2].tonumpy()
        self.assertTrue(np.allclose(expected, val))

