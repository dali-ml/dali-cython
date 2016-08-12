import sys
sys.path.append("..")
import unittest

import numpy as np
import dali

class LSTMTests(unittest.TestCase):
    def test_construction(self):

        for num_children in range(5):
            for memory_feeds_gates in [True, False]:
                params_per_stacked_layer = 2 + num_children


                lstm = dali.layers.LSTM(1, 2, num_children=num_children, memory_feeds_gates=memory_feeds_gates)
                self.assertEqual(num_children, lstm.num_children)
                self.assertEqual(1, lstm.input_size)
                self.assertEqual(2, lstm.hidden_size)
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
