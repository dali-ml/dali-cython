import numpy as np

import dali.core as D
import dali

from dali import beam_search

class TranslationModel(object):
    def __init__(self, input_size, hiddens,
                       encoder_vocab_size, decoder_vocab_size,
                       softmax_input_size=None, dtype=np.float32):
        self.input_size = input_size
        self.hiddens    = hiddens
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.softmax_input_size = softmax_input_size
        self.dtype = dtype

        self.encoder_embedding = D.random.uniform(-0.05, 0.05, (encoder_vocab_size, input_size), dtype=dtype)
        self.decoder_embedding = D.random.uniform(-0.05, 0.05, (decoder_vocab_size, input_size), dtype=dtype)

        self.encoder_lstm    = D.StackedLSTM(input_size, hiddens, memory_feeds_gates=True, dtype=dtype)
        self.decoder_lstm    = D.StackedLSTM(input_size, hiddens, memory_feeds_gates=True, dtype=dtype)

        if self.softmax_input_size is not None:
            self.predecoder = D.StackedInputLayer(self.hiddens, self.softmax_input_size)
            self.decoder = D.Layer(self.softmax_input_size, decoder_vocab_size, dtype=dtype)
        else:
            self.decoder = D.Layer(hiddens[-1], decoder_vocab_size, dtype=dtype)

    def decode_state(self, state):
        if self.softmax_input_size is not None:
            decoder_input = self.predecoder.activate([s.hidden for s in state])
        else:
            decoder_input = state[-1].hidden
        return self.decoder.activate(decoder_input)

    def error(self, batch):
        error = D.Mat(1,1)
        state = self.encoder_lstm.initial_states()
        for ts in range(batch.timesteps):
            inputs  = batch.inputs(ts)
            targets = batch.targets(ts)
            if ts < batch.from_timesteps:
                assert targets is None
                encoded = self.encoder_embedding[inputs]
                state = self.encoder_lstm.activate(encoded, state)
            else:
                assert inputs is None
                decoded = self.decode_state(state)
                # mask the error - only for the relevant sentences
                tstep_error = batch.masks(ts).T() * D.MatOps.softmax_cross_entropy(decoded, targets)
                #tstep_error = D.MatOps.softmax_cross_entropy(decoded, targets)
                error = error + tstep_error.sum()
                # feedback the predictions
                if ts + 1 != batch.timesteps:
                    # for the last timestep encoding is not necessary
                    encoded = self.decoder_embedding[targets]
                    state = self.decoder_lstm.activate(encoded, state)

        return error

    def predict(self, input_sentence, **kwargs):
        with D.NoBackprop():
            state = self.encoder_lstm.initial_states()
            for word_idx in input_sentence:
                encoded = self.encoder_embedding[word_idx]
                state = self.encoder_lstm.activate(encoded, state)
            def candidate_scores(state):
                decoded = self.decode_state(state)
                return D.MatOps.softmax(decoded).log()
            def make_choice(state, candidate_idx):
                encoded = self.decoder_embedding[candidate_idx]
                return self.decoder_lstm.activate(encoded, state)

            return beam_search(state,
                               candidate_scores,
                               make_choice,
                               **kwargs)

    def parameters(self):
        ret = ([self.encoder_embedding,
               self.decoder_embedding]
            + self.encoder_lstm.parameters()
            + self.decoder_lstm.parameters()
            + self.decoder.parameters())
        if self.softmax_input_size is not None:
            ret.extend(self.predecoder.parameters())
        return ret

    def name_parameters(self, prefix):
        self.encoder_embedding.name = prefix + ".encoder_embedding"
        self.decoder_embedding.name = prefix + ".decoder_embedding"
        self.encoder_lstm.name_parameters(prefix + ".encoder_lstm")
        self.decoder_lstm.name_parameters(prefix + ".decoder_lstm")
        self.decoder.name_parameters(prefix + ".decoder")
        if self.softmax_input_size is not None:
            self.predecoder.name_parameters(prefix + ".predecoder")
