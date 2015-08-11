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

START_TOKEN = '**START**'

def create_lines_batch(lines, vocab, add_start_token=False, fill_eos=False, add_eos=False):
    encoded_lines = []
    for l in lines:
        if type(l) == str:
            l = l.split(" ")
        if add_start_token:
            l = [START_TOKEN] + l
        encoded_lines.append(vocab.encode(l, add_eos=add_eos))

    seq_length = max(map(len, encoded_lines))
    # we add one index to account for start of sequence token
    data = np.empty((seq_length, len(lines)), dtype=np.int32)
    if fill_eos:
        data.fill(vocab.eos)
    else:
        data.fill(0)

    for line_idx, encoded_line in enumerate(encoded_lines):
        data[:len(encoded_line), line_idx] = encoded_line
    data = D.Mat(data, borrow=True, dtype=np.int32)
    return data

class LMBatch(object):
    @staticmethod
    def given_vocab(vocab, **kwargs):
        def wrapper(sentences):
            return LMBatch(sentences, vocab, **kwargs)
        return wrapper

    def __init__(self, sentences, vocab, store_originals=False, fill_eos=True, add_eos=True, add_start_token=True):
        if store_originals:
            self.sentences = sentences
        self.sentence_lengths = [len(s) for s in sentences]
        self.data = create_lines_batch(
            sentences,
            vocab,
            add_start_token=add_start_token,
            fill_eos=fill_eos,
            add_eos=add_eos
        )
        self.timesteps = self.data.shape[0] - 1
        self.examples  = self.data.shape[1]

    def inputs(self, timestep):
        return self.data[timestep]

    def targets(self, timestep):
         # predictions are offset by 1 to inputs, so
        return self.data[timestep + 1]
