import sys
sys.path.append("..")
import unittest

import numpy as np
import dali

class LSTMTests(unittest.TestCase):
    def test_construction(self):

        for dtype in [np.float32, np.float64]:
            lstm = dali.layers.LSTM(1, 2, dtype=dtype)
            self.assertEqual(dtype, lstm.dtype)

        for num_children in range(5):
            for memory_feeds_gates in [True, False]:
                params_per_stacked_layer = 2 + num_children


                lstm = dali.layers.LSTM(1, 2, num_children=num_children, memory_feeds_gates=memory_feeds_gates)
                self.assertEqual(num_children, lstm.num_children)
                self.assertEqual(1, lstm.input_size)
                self.assertEqual(2, lstm.hidden_size)
                self.assertEqual(num_children, len(lstm.forget_layers))
                params = lstm.parameters()

                num_params = (
                    3 * params_per_stacked_layer +
                    num_children * params_per_stacked_layer
                )
                if memory_feeds_gates:
                    num_params = (
                        num_params +
                        1 + #Wco
                        num_children * 2 # Wcells
                    )
                self.assertEqual(num_params, len(params))

    def test_activation(self):
        ipt_amount = 2.5
        cell_write_amount = 0.45
        output_gate_amount = 1.0

        lstm = dali.layers.LSTM(1, 2)
        state = lstm.initial_states()
        self.assertEqual(state.memory.shape, (1, 2))
        self.assertEqual(state.hidden.shape, (1, 2))

        lstm.input_layer.tensors[0].w.get_value().fill(0.0)
        lstm.input_layer.b.w.get_value().fill(1000.0)

        lstm.output_layer.tensors[0].w.get_value().fill(0.0)
        lstm.output_layer.b.w.get_value().fill(output_gate_amount)

        lstm.cell_layer.tensors[0].w.get_value()[0, 0] = cell_write_amount
        lstm.cell_layer.tensors[0].w.get_value()[0, 1] = 0.0
        lstm.cell_layer.b.w.get_value().fill(0.0)

        res = lstm.activate(dali.Tensor([[ipt_amount]], dtype=lstm.dtype), state)

        self.assertEqual(res.hidden.shape, (1, 2))
        self.assertEqual(res.memory.shape, (1, 2))

        self.assertTrue(
            abs(
                res.memory.w.get_value()[0,0] -
                np.tanh(cell_write_amount * ipt_amount)
            ) < 1e-6
        )

        self.assertTrue(
            abs(
                res.hidden.w.get_value()[0,0] -
                np.tanh(
                    np.tanh(cell_write_amount * ipt_amount)
                ) * 1.0 / (1.0 + np.exp(-output_gate_amount))
            ) < 1e-6
        )
