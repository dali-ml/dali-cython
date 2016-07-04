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

def create_lines_batch(lines, vocab, add_start_token=False, fill_eos=False, add_eos=False, align_right=False):
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
        if align_right:
            data[-len(encoded_line):, line_idx] = encoded_line
        else:
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


class TranslationBatch(object):
    @staticmethod
    def given_vocabs(vocabs, **kwargs):
        def wrapper(sentence_pairs):
            return TranslationBatch(sentence_pairs, vocabs, **kwargs)
        return wrapper


    def __init__(self, sentence_pairs, vocabs, store_originals=False, input_add_eos=False, output_add_eos=True, add_start_token=False, reverse_input=True):
        if store_originals:
            self.sentence_pairs = sentence_pairs
        from_sentences = [sentence_pair[0] for sentence_pair in sentence_pairs]
        to_sentences   = [sentence_pair[1] for sentence_pair in sentence_pairs]

        if reverse_input:
            from_sentences = [list(reversed(s)) for s in from_sentences]


        from_vocab, to_vocab = vocabs

        from_eos_correction = (1 if input_add_eos else 0)
        to_eos_correction   = (1 if output_add_eos else 0)

        self.from_data = create_lines_batch(
            from_sentences,
            from_vocab,
            add_start_token=add_start_token,
            fill_eos=False,
            add_eos=input_add_eos,
            align_right=True
        )

        self.to_data = create_lines_batch(
            to_sentences,
            to_vocab,
            add_start_token=add_start_token,
            fill_eos=True,
            add_eos=output_add_eos
        )
        self.from_tokens  = sum(map(len, from_sentences)) + from_eos_correction * len(from_sentences)
        self.to_tokens    = sum(map(len, to_sentences))   + to_eos_correction  * len(to_sentences)

        self.target_mask = D.Mat(*self.to_data.shape)
        for example_idx, sentence in enumerate(to_sentences):
            for ts in range(len(sentence) + to_eos_correction):
                self.target_mask.w[ts, example_idx] = 1
                self.target_mask.constant = True

        self.from_timesteps = self.from_data.shape[0]
        self.timesteps = self.from_data.shape[0] + self.to_data.shape[0]
        self.examples  = len(sentence_pairs)

    def inputs(self, timestep):
        if timestep < self.from_timesteps:
            return self.from_data[timestep]
        else:
            return None

    def targets(self, timestep):
        if timestep >= self.from_timesteps:
            return self.to_data[timestep - self.from_timesteps]
        else:
            return None

    def masks(self, timestep):
        if timestep >= self.from_timesteps:
            return self.target_mask[timestep - self.from_timesteps]
        else:
            return None
