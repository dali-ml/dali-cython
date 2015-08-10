import unittest

from dali.utils import Vocab, VocabEncoded

class VocabTests(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.vocab = Vocab()
        self.vocab.add([[{
            'interesting_words': ['awesome', 'cat', 'lol'],
            'daniel' : 'daniel',
            'wtf':[[[[[[[[[[[['there']]]]]]]]]]]]
        }]])

        self.example =  {1:{1:{1:[[[[[ 'awesome', 'but','staph', 'daniel' ]]]]]}}}
        self.example_unks = {1: {1: {1: [[[[['awesome', '**UNK**', '**UNK**', 'daniel']]]]]}}}


    def test_addition(self):
        self.assertEqual(set(self.vocab.words()),
                         set(['awesome',
                              'there',
                              'daniel',
                              '**UNK**',
                              'cat',
                              '**EOS**',
                              'lol']))


    def test_encode(self):
        encoded  = self.vocab.encode(self.example)
        decoded  = self.vocab.decode(encoded, decode_type=VocabEncoded)
        self.assertEqual(self.example_unks, decoded)

    def test_encode_eos(self):
        encoded  = self.vocab.encode(self.example, add_eos=True)
        decoded  = self.vocab.decode(encoded, strip_eos=True, decode_type=VocabEncoded)
        assert self.example_unks == decoded
