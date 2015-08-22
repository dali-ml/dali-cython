from collections import defaultdict

from .batch import TranslationBatch
from .process import Process, Multiplexer, Lines, DiscoverFiles, IdentityReducer, BatchBenefactor
from dali.utils import Vocab


class TranslationFiles(object):
    def __init__(self, root_path, from_lang, to_lang):
        files_from = set(DiscoverFiles(root_path, "." + from_lang))
        files_to   = set(DiscoverFiles(root_path, "." + to_lang))

        self.pairs = []
        for file_name in files_from:
            pref = file_name[:-(len(from_lang) + 1)]
            hypothetical_to_file = pref + '.' + to_lang
            if hypothetical_to_file in files_to:
                self.pairs.append((file_name, hypothetical_to_file))

        self.next_pair = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_pair >= len(self.pairs):
            raise StopIteration()
        else:
            self.next_pair += 1
            return self.pairs[self.next_pair - 1]


def TranslationMapper(sentence_bounds=(None, None)):
    def translation_lines():
        lines =  Lines()                         \
                 .split_punctuation()            \
                 .split_spaces()                 \
                 .bound_length(*sentence_bounds)
        return lines


    return Multiplexer(translation_lines(), translation_lines())


def build_vocabs(path, from_lang, to_lang, from_max_size=None, to_max_size=None):
    from_occurence = defaultdict(lambda: 0)
    to_occurence   = defaultdict(lambda: 0)
    try:
        for from_sentence, to_sentence in Process(files=TranslationFiles(path, from_lang, to_lang),
                                                  mapper=TranslationMapper(sentence_bounds=(None, None)),
                                                  reducer=IdentityReducer()):
            for word in from_sentence:
                from_occurence[word] += 1

            for word in to_sentence:
                to_occurence[word] += 1
    except KeyboardInterrupt:
        print('Impatient User Detected, file processing halted, proceeding to build vocab.')


    from_occurence = list(from_occurence.items())
    to_occurence   = list(to_occurence.items())

    # highest occurrence first
    from_occurence.sort(key=lambda x: x[1], reverse=True)
    to_occurence  .sort(key=lambda x: x[1], reverse=True)

    # remove occurences, keep sorted words
    from_occurence = [x[0] for x in from_occurence]
    to_occurence =   [x[0] for x in to_occurence]

    from_vocab = Vocab(from_occurence[:from_max_size])
    to_vocab   = Vocab(to_occurence[:to_max_size])

    return from_vocab, to_vocab

def iterate_examples(root_path, from_lang, to_lang, vocabs, minibatch_size, reverse_input=True, sentences_until_minibatch=None, sentence_length_bounds=(None, None)):
    sentences_until_minibatch = sentences_until_minibatch or 10000 * minibatch_size
    files   = TranslationFiles(root_path, from_lang, to_lang)
    mapper = TranslationMapper(sentence_bounds=sentence_length_bounds)
    sorting_key = lambda sentence_pair: (len(sentence_pair[0]), len(sentence_pair[1])) # sort by length of the input sentence first and then by the length of the output sentence

    reducer = BatchBenefactor(minibatch_size,
                              TranslationBatch.given_vocabs(vocabs, store_originals=True, reverse_input=reverse_input),
                              sentences_until_minibatch,
                              sorting_key=sorting_key)
    return Process(files=files, mapper=mapper, reducer=reducer)
