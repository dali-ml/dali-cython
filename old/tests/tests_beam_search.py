import math
import unittest

import numpy as np
import dali.core as D

from dali import beam_search, Beam

class BeamSearchTests(unittest.TestCase):
    def test_letters(self):
        MAX_LENGTH = 2
        choices = {
            #initial_choices
            "a": 0.6,
            "b": 0.4,
            #after chosing a
            "aa": 0.55,  # (total worth 0.33)
            "ab": 0.45,  # (total worth 0.18)
            #after choosing b
            "ba": 0.99,  # (total worth 0.495)
            "bb": 0.11,  # (total worth 0.044)
        };

        # Above example is designed to demonstrate greedy solution,
        # as well as better optimal solution:
        # GREEDY:    (beam_width == 1) => "aa" worth 0.33
        # OPTIMAL:   (beam_width == 2) => "ba" worth 0.495
        res_aa = Beam([0,0], D.Mat([math.log(0.6 * 0.55)]), "aa")
        res_ab = Beam([0,1], D.Mat([math.log(0.6 * 0.45)]), "ab")
        res_ba = Beam([1,0], D.Mat([math.log(0.4 * 0.99)]), "ba")
        res_bb = Beam([1,1], D.Mat([math.log(0.4 * 0.11)]), "bb")

        initial_state = "";
        def candidate_scores(state):
            ret = D.Mat(1,2)
            ret.w[0,0] = math.log(choices[state + "a"])
            ret.w[0,1] = math.log(choices[state + "b"])
            return ret
        def make_choice(prev_state, choice):
            return prev_state + ("a" if choice == 0 else "b")

        def my_beam_search(beam_width):
            return beam_search(initial_state=initial_state,
                               candidate_scores=candidate_scores,
                               make_choice=make_choice,
                               beam_width=beam_width,
                               max_sequence_length = MAX_LENGTH)

        def beams_equal(b1, b2):
            return (b1.solution == b2.solution and
                    np.allclose(b1.score.w, b2.score.w) and
                    b1.state == b2.state)

        def results_equal(a,b):
            return len(a) == len(b) and all(beams_equal(b1,b2) for b1,b2 in zip(a,b))

        with self.assertRaises(ValueError):
            my_beam_search(0)

        self.assertTrue(results_equal(my_beam_search(1), [res_aa]))
        self.assertTrue(results_equal(my_beam_search(2), [res_ba, res_aa]))
        self.assertTrue(results_equal(my_beam_search(4), [res_ba, res_aa, res_ab, res_bb]))
        self.assertTrue(results_equal(my_beam_search(10),[res_ba, res_aa, res_ab, res_bb]))

if __name__ == '__main__':
    unittest.main()
