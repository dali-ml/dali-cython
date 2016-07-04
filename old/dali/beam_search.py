from collections import namedtuple
from types import FunctionType
import dali.core as D

Beam = namedtuple("Beam", ["solution", "score", "state"])

def beam_search(initial_state,
                candidate_scores,
                make_choice,
                beam_width=5,
                eos_symbol = None,
                max_sequence_length=None,
                blacklist=[]):

    if beam_width <= 0:
        raise ValueError("Beam width must be positive, received: " + str(beam_width))

    iterations = 0
    results = [
        Beam([], D.Mat(1,1), initial_state)
    ]

    def lazy_beam(prev_beam, candidate, new_score):
        def generate():
            choice = make_choice(prev_beam.state, candidate)
            if type(choice) == tuple:
                choice_repr, new_state  = choice
            else:
                choice_repr, new_state = candidate, choice
            return Beam(
                prev_beam.solution + [choice_repr],
                new_score,
                new_state,
            )
        return generate

    def lazy_identity(beam):
        def generate():
            return beam
        return generate

    if eos_symbol is None:
        check_eos = lambda beam: False
    elif type(eos_symbol) != FunctionType:
        check_eos = lambda beam: len(beam.solution) > 0 and beam.solution[-1] == eos_symbol
    else:
        check_eos = eos_symbol

    while max_sequence_length is None or iterations < max_sequence_length:
        proposals = []
        for beam in results:
            if check_eos(beam):
                proposals.append((beam.score, lazy_identity(beam)))
            else:
                scores = candidate_scores(beam.state)
                sorted_candidates = D.MatOps.argsort(scores)

                sorted_candidates = sorted_candidates[::-1]
                candidates_remaining = beam_width
                for candidate_idx in sorted_candidates:
                    if candidate_idx in blacklist:
                        continue
                    new_score = beam.score + scores.T()[candidate_idx]
                    proposals.append((new_score, lazy_beam(beam, candidate_idx, new_score)))
                    candidates_remaining -= 1
                    if candidates_remaining <= 0:
                        break
        proposals.sort(reverse=True, key=lambda x: x[0].w[0])
        results = [ eval_beam() for _, eval_beam in proposals[:beam_width]]

        iterations += 1

    return results

__all__ = ["beam_search", "Beam"]
