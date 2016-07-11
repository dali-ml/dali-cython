from libcpp.string   cimport string
from .cython_utils cimport normalize_s

from third_party.libcpp11.unordered_map cimport unordered_map
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/utils/vocab.h" namespace "utils" nogil:

    const char* c_end_symbol "utils::end_symbol"

    cdef cppclass CVocab "utils::Vocab":
        unordered_map[string, int] word2index
        vector[string] index2word
        size_t size() const
        int unknown_word
        CVocab()
        CVocab(vector[string]&)
        CVocab(vector[string]&, bint unknown_word)

cdef vector[string] list_string_to_vector(list words)
cdef list vector_string_to_list(vector[string]& words)
cdef dict unordered_map_string_to_dict(unordered_map[string, int]& word2index)
cdef unordered_map[string, int] dict_string_to_unordered_map(dict word2index)

cdef class Vocab:
    """
    Vocab()

    Keep a mapping between integer ids and strings
    to encode and decode character sequences into
    labels or one-hot inputs.

    Vocab.__init__(self, words=[], unknown_word=True)

    Parameters
    ----------
    words : list of strings in the order they should
            be indexed.
    unknown_word : bool, whether a special out of vocabulary
                   word should be added and stored in the
                   vocabulary as the last word.
    """
    cdef CVocab vocabinternal
