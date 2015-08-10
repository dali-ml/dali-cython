import numpy as np

import dali.core as D

class Batch(object):
    def __init__(self):
        self.timesteps = 0
        self.examples  = 0
    def inputs(timestep):
        return None
    def targets(timestep):
        return None

    def __repr__(self):
        return 'Batch(timesteps=%d, examples=%d)' % (self.timesteps, self.examples)

class LMBatch(object):
    START_TOKEN = '**START**'
    @staticmethod
    def given_vocab(vocab, **kwargs):
        def wrapper(sentences):
            return LMBatch(sentences, vocab, **kwargs)
        return wrapper

    def __init__(self, sentences, vocab, store_originals=False, add_eos=True):
        if store_originals:
            self.sentences = sentences
        sentences = [vocab.encode(s, add_eos=add_eos) for s in sentences]

        self.sentence_lengths = [len(s) for s in sentences]

        self.timesteps = max(self.sentence_lengths)
        self.examples  = len(sentences)
        # we add one index to account for start of sequence token
        self.data = np.empty((self.timesteps + 1, self.examples))
        # data is badded by EOS
        self.data.fill(vocab.eos)
        self.data[0,:].fill(vocab[LMBatch.START_TOKEN])
        for example_idx, example in enumerate(sentences):
            self.data[1:(len(example) + 1), example_idx] = example
        self.data = D.Mat(self.data, dtype=np.int32)

    def inputs(self, timestep):
        return self.data[timestep]

    def targets(self, timestep):
         # predictions are offset by 1 to inputs, so
        return self.data[timestep + 1]
