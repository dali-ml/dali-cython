import types

def apply_recursively_on_type(x, f, target_type, list_callback=None):
    if type(x) == target_type:
        return f(x)
    elif type(x) == list or isinstance(x, types.GeneratorType):
        ret = [ apply_recursively_on_type(el, f, target_type, list_callback) for el in x]
        if list_callback and all(type(el) == target_type for el in x):
            ret = list_callback(ret)
        return ret
    elif type(x) == dict:
        res = {}
        for k,v in x.items():
            res[k] = apply_recursively_on_type(v, f, target_type, list_callback)
        return res
    else:
        return x

class Vocab(object):
    UNK = '**UNK**'
    EOS = '**EOS**'

    def __init__(self, words=None, add_eos=True, add_unk=True):
        self.index2word = []
        self.word2index = {}
        self.eos = None
        self.unk = None
        if add_unk:
            self.add(Vocab.UNK)
        if add_eos:
            self.add(Vocab.EOS)

        if words:
            self.add(words)


    def __contains__(self, key):
        if isinstance(key, int):
            return key in range(len(self.index2word))
        elif isinstance(key, str):
            return key in self.word2index
        else:
            raise ValueError("expected(index or string)")

    def add(self, obj):
        def add_f(word):
            idx = self.word2index.get(word)
            if idx is None:
                idx = len(self.index2word)
                self.index2word.append(word)
                self.word2index[word] = idx
                if word is Vocab.UNK:
                    self.unk = idx
                if word is Vocab.EOS:
                    self.eos = idx
            return word
        apply_recursively_on_type(obj, add_f, str)

    def words(self):
        return self.word2index.keys()

    def __len__(self):
        return len(self.index2word)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.index2word[index]
        elif isinstance(index, str):
            if self.unk is not None:
                return self.word2index.get(index) or self.unk
            else:
                return self.word2index[index]
        else:
            raise ValueError("expected(index or string)")

    def decode(self, obj, strip_eos=False, decode_type=int):
        def decode_f(word_idx):
            return self.index2word[word_idx]
        def decode_list_f(lst):
            if strip_eos:
                assert self.eos is not None
                return [el for el in lst if el != Vocab.EOS]
            else:
                return lst
        return apply_recursively_on_type(obj, decode_f, decode_type, list_callback=decode_list_f)

    def encode(self, obj, add_eos=False, encode_type=int):
        def encode_f(word):
            if self.unk is not None:
                return encode_type(self.word2index.get(word) or self.unk)
            else:
                return encode_type(self.word2index[word])
        def encode_list_f(lst):
            lst = [encode_f(word) for word in lst]
            if add_eos:
                assert self.eos is not None
                lst.append(self.eos)
            return lst
        return apply_recursively_on_type(obj, lambda x:x, str, list_callback=encode_list_f)

__all__ = [
    "Vocab"
]
