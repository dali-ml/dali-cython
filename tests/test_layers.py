import sys
import pickle

sys.path.append("..")
import unittest

import numpy as np
import dali

class LayersTests(unittest.TestCase):
    def test_layers_pickle(self):
            nets = [
                dali.layers.Layer(2, 3, dtype=np.float64),
                dali.layers.StackedInputLayer([1,2], 3, dtype=np.float64)
            ]

            for net in nets:
                saved = pickle.dumps(net)
                loaded = pickle.loads(saved)
                self.assertEqual(len(net.parameters()), len(loaded.parameters()))
                for param, lparam in zip(net.parameters(), loaded.parameters()):
                    self.assertTrue(np.allclose(param.w.get_value(), lparam.w.get_value(), atol=1e-6))
