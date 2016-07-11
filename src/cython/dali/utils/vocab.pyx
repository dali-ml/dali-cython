# source : https://github.com/cython/cython/blob/master/Cython/Includes/libcpp/unordered_map.pxd
end_symbol = c_end_symbol.decode("utf-8")

cdef vector[string] list_string_to_vector(list words):
    cdef vector[string] out
    for word in words:
        out.push_back(word.encode("utf-8"))
    return out

cdef list vector_string_to_list(vector[string]& words):
    out = []
    for word in words:
        out.append(word.decode("utf-8"))
    return out

cdef dict unordered_map_string_to_dict(unordered_map[string, int]& word2index):
    word2index_dict = {}
    for pair in word2index:
        word2index_dict[pair.first.decode("utf-8")] = pair.second
    return word2index_dict

cdef unordered_map[string, int] dict_string_to_unordered_map(dict word2index):
    cdef unordered_map[string, int] word2index_map
    for key, value in word2index.items():
        word2index_map[key.encode("utf-8")] = value
    return word2index_map

cdef class Vocab:
    def __cinit__(self, list words=[], bint unknown_word=True):
        self.vocabinternal = CVocab(list_string_to_vector(words), unknown_word)

    def __len__(self):
        return self.vocabinternal.size()

    def __getitem__(self, index):
        if type(index) == int:
            word = self.vocabinternal.index2word.at(index)
            return word.decode("utf-8")
        elif type(index) == str:
            return self.vocabinternal.word2index.at(index.encode("utf-8"))
        else:
            raise TypeError("Vocab only get items using int or str as keys.")

    def __setitem__(self, index, value):
        if type(index) == int:
            if index < 0:
                raise IndexError("list index must be positive")
            if index >= self.vocabinternal.index2word.size():
                raise IndexError("list index out of range")
            self.vocabinternal.index2word[index] = value.encode("utf-8")
        elif type(index) == str:
            self.vocabinternal.word2index[index.encode("utf-8")] = value
        else:
            raise TypeError("Vocab only sets items using int or str as keys.")

    property unknown_word:
        def __get__(self):
            return self.vocabinternal.unknown_word

        def __set__(self, int value):
            self.vocabinternal.unknown_word = value

    property index2word:
        def __get__(self):
            return vector_string_to_list(self.vocabinternal.index2word)

        def __set__(self, list words):
            self.vocabinternal.index2word = list_string_to_vector(words)

    property word2index:
        def __get__(self):
            return unordered_map_string_to_dict(self.vocabinternal.word2index)

        def __set__(self, dict words):
            self.vocabinternal.word2index = dict_string_to_unordered_map(words)
