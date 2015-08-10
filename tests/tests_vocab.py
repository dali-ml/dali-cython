import unittest

from dali.utils import Vocab, VocabEncoded

class VocabTests(unittest.TestCase):
    def test_vocab(self):
        vocab = Vocab()
        vocab.add([[{
            'interesting_words': ['awesome', 'cat', 'lol'],
            'daniel' : 'daniel',
            'wtf':[[[[[[[[[[[['there']]]]]]]]]]]]
        }]])
        assert(set(vocab.words()) == set(['awesome', 'there', 'daniel', '**UNK**', 'cat', '**EOS**', 'lol']))
        original = {1:{1:{1:[[[[[ 'awesome', 'but','staph', 'daniel' ]]]]]}}}
        original_with_unks = {1: {1: {1: [[[[['awesome', '**UNK**', '**UNK**', 'daniel']]]]]}}}
        encoded  = vocab.encode(original)
        decoded  = vocab.decode(encoded, decode_type=VocabEncoded)
        assert original_with_unks == decoded

        encoded  = vocab.encode(original, add_eos=True)
        decoded  = vocab.decode(encoded, strip_eos=True, decode_type=VocabEncoded)
        assert original_with_unks == decoded
