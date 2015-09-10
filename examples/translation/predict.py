import argparse
import random
import math

from dali.utils import (
    set_device_from_args,
    add_device_args,
    unpickle_as_dict,
)
from dali.data.utils import split_punctuation

from translation import TranslationModel

def parse_args():
    parser = argparse.ArgumentParser()
    add_device_args(parser)
    parser.add_argument("--path",              type=str,  required='True',  help="Path to saved model")
    parser.add_argument("--beam_width",        type=int,  default=5,        help="Beam width used when prediction")
    parser.add_argument("--max_output_length", type=int,  default=40,       help="Maximum number of words in the translation")
    parser.add_argument("--show_beams",        action='store_true', default=False,
                                               help="If true shows all the beams and probabilities")

    return parser.parse_args()

def show_reconstructions(model, example_pair, vocabs, max_sentence_length):
    from_words, to_words = example_pair
    from_vocab, to_vocab = vocabs
    from_with_unk = ' '.join(from_vocab.decode(from_vocab.encode(from_words)))
    to_with_unk   = ' '.join(to_vocab.decode(to_vocab.encode(to_words)))
    print('TRANSLATING: %s' % from_with_unk)
    print('REFERENCE:   %s' % to_with_unk)
    print('')


def main(args):
    set_device_from_args(args)

    RELEVANT_VARIABLES = ["model", "vocabs"]
    loaded = unpickle_as_dict(args.path, RELEVANT_VARIABLES)
    model = loaded["model"]
    from_vocab, to_vocab = loaded["vocabs"]

    while True:
        from_sentence = split_punctuation(input()).split(' ')
        encoded       = from_vocab.encode(list(reversed(from_sentence)), add_eos=False)

        beams = model.predict(encoded,
                              eos_symbol=to_vocab.eos,
                              max_sequence_length=args.max_output_length + 1,
                              beam_width=args.beam_width)

        if args.show_beams:
            for solution, score, _ in beams:
                score = math.exp(score.w[0])
                # reveal the unks
                solution = ' '.join(to_vocab.decode(solution, strip_eos=True))
                print('%f => %s' % (score, to_vocab.decode(solution, True)))
        else:
            print(' '.join(to_vocab.decode(beams[0].solution, strip_eos=True)))



if __name__ == '__main__':
    main(parse_args())
