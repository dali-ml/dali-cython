import sys
import pickle
sys.path.append("..")
import unittest

import numpy as np
import dali

def reference_sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.0)

def reference_relu(x):
    return np.maximum(x, 0.0)

def reference_softplus(x):
    return (x > 20) * x + (x <= 20) * np.log(np.exp(x) + 1)

def reference_prelu(x, weights):
    return (x > 0) * x + (x <= 0) * x * weights

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
            expected = reference_op(t.w.get_value())
            self.assertTrue(np.allclose(expected, op(t).w.get_value(), atol=1e-6))

    def test_binary_add(self):
        left = dali.Tensor.uniform(-2.0, 2.0, (2, 3))
        right = dali.Tensor.uniform(-2.0, 2.0, (2, 3))
        expected = left.w.get_value() + right.w.get_value()
        res = dali.tensor.op.binary.add(left, right)
        self.assertTrue(np.allclose(expected, res.w.get_value(), atol=1e-6))

    def test_binary_add_n(self):
        left = dali.Tensor.uniform(-2.0, 2.0, (2, 3))
        middle = dali.Tensor.uniform(-2.0, 2.0, (2, 3))
        right = dali.Tensor.uniform(-2.0, 2.0, (2, 3))
        expected = left.w.get_value() + middle.w.get_value() + right.w.get_value()
        res = dali.tensor.op.binary.add_n((left, middle, right))
        self.assertTrue(np.allclose(expected, res.w.get_value(), atol=1e-6))

    def test_binary_prelu(self):
        x = dali.Tensor.uniform(-2.0, 2.0, (2, 3))
        weights = dali.Tensor.uniform(0.1, 2.0, (2, 3))
        expected = reference_prelu(x.w.get_value(), weights.w.get_value())
        res = dali.tensor.op.binary.prelu(x, weights)
        self.assertTrue(np.allclose(expected, res.w.get_value(), atol=1e-6))

    def test_composite_quadratic_form(self):
        left = dali.Tensor.uniform(-20.0, 20.0, (2, 4))
        middle = dali.Tensor.uniform(-20.0, 20.0, (2, 3))
        right = dali.Tensor.uniform(-20.0, 20.0, (3, 5))

        expected = (left.w.get_value().T.dot(middle.w.get_value())).dot(right.w.get_value())
        res = dali.tensor.op.composite.quadratic_form(left, middle, right)
        self.assertTrue(np.allclose(expected, res.w.get_value(), atol=1e-6))

    def test_dot_tensordot(self):
        # this is the same example as given in the numpy
        # documentation:
        a = dali.Tensor.arange(60.).reshape(3,4,5)
        b = dali.Tensor.arange(24.).reshape(4,3,2)
        c = dali.tensor.op.dot.tensordot(a, b, axes=([1,0],[0,1]))
        self.assertEqual((5, 2), c.shape)
        self.assertTrue(
            np.allclose(
                [[ 4400.,  4730.],
                 [ 4532.,  4874.],
                 [ 4664.,  5018.],
                 [ 4796.,  5162.],
                 [ 4928.,  5306.]],
                c.w.get_value(),
                atol=1e-6
            )
        )

    def test_tape(self):
        dali.tensor.tape.clear()
        self.assertEqual(dali.tensor.tape.size(), 0)
        a = dali.Tensor.zeros((2, 1))
        b = a.sigmoid()
        self.assertEqual(dali.tensor.tape.size(), 1)

    def test_tensor_pickle(self):
        tensor = dali.Tensor((0.5, 0.5, 0.9999,), dtype=np.float64)
        saveable = pickle.dumps(tensor)
        loaded = pickle.loads(saveable)
        self.assertTrue(np.allclose(tensor.w.tonumpy(), loaded.w.tonumpy()))
        self.assertEqual(tensor.dw.shape, loaded.dw.shape)
