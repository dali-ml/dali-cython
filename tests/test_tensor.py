import sys
sys.path.append("..")
import unittest

import numpy as np
import dali

def reference_sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.0)

def reference_relu(x):
    return np.maximum(x, 0.0)

def reference_softplus(x):
    return (
        (x > 20.0) * x +
        (x <= 20.0) * np.log(1.0 + np.exp(x))
    )

class TensorTests(unittest.TestCase):
    def test_swapaxes(self):
        t = dali.Tensor.zeros((1, 2, 3, 4))
        t = t.swapaxes(0, -1)
        self.assertEqual((4, 2, 3, 1), t.shape)

    def test_unary_op(self):
        ops = [
            dali.tensor.op.unary.sigmoid,
            dali.tensor.op.unary.tanh,
            dali.tensor.op.unary.relu,
            dali.tensor.op.unary.softplus
        ]
        reference_ops = [
            reference_sigmoid,
            np.tanh,
            reference_relu,
            reference_softplus
        ]

        for op, reference_op in zip(ops, reference_ops):
            t = dali.Tensor.uniform(-2.0, 2.0, (2, 3))
            t_np = t.w.get_value(copy=False)
            res = op(t)
            self.assertEqual(t.shape, res.shape)
            self.assertTrue(np.allclose(reference_op(t_np), res.w.get_value(), atol=1e-6))

