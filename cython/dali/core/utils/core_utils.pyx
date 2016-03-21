from libcpp11.unordered_map cimport unordered_map

# source : https://github.com/cython/cython/blob/master/Cython/Includes/libcpp/unordered_map.pxd

cdef extern from "core/utils/cpp_utils.h" nogil:
    void print_str(string)

cdef extern from "dali/utils/random.h" namespace "utils::random" nogil:
    void reseed()
    void set_seed(int)

cdef extern from "dali/utils/random.h" namespace "utils" nogil:
    double randdouble(double, double)
    int randint(int,int)

cdef extern from "dali/utils/core_utils.h" namespace "utils" nogil:
    string cpp_trim "utils::trim" (string)
    string cpp_ltrim "utils::ltrim" (string)
    string cpp_rtrim "utils::rtrim" (string)

    const char* c_end_symbol "utils::end_symbol"

    cdef cppclass CVocab "utils::Vocab":
        unordered_map[string, unsigned int] word2index
        vector[string] index2word
        size_t size() const
        unsigned int unknown_word
        CVocab()
        CVocab(vector[string]&)
        CVocab(vector[string]&, bint unknown_word)

class utils:
    end_symbol = c_end_symbol.decode("utf-8")

    @staticmethod
    def trim(s):
        """Trim a string, remove whitespace on either side"""
        cdef string s_norm = normalize_s(s)
        return cpp_trim(s_norm)

    @staticmethod
    def randint(int low=0, int high=1):
        cdef int out
        with nogil:
            out = randint(low,high)
        return out

    @staticmethod
    def randdouble(float low=0.0, float high=1.0):
        cdef double out
        with nogil:
            out = randdouble(low,high)
        return out

    @staticmethod
    def reseed():
        with nogil:
            reseed()

    @staticmethod
    def set_seed(int newseed):
        with nogil:
            set_seed(newseed)

    @staticmethod
    def rtrim(s):
        """Trim right-side of a string."""
        cdef string s_norm = normalize_s(s)
        return cpp_rtrim(s_norm)

    @staticmethod
    def ltrim(s):
        """Trim left-side of a string."""
        cdef string s_norm = normalize_s(s)
        return cpp_ltrim(s_norm)

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

cdef dict unordered_map_string_to_dict(unordered_map[string, unsigned int]& word2index):
    word2index_dict = {}
    for pair in word2index:
        word2index_dict[pair.first.decode("utf-8")] = pair.second
    return word2index_dict

cdef unordered_map[string, unsigned int] dict_string_to_unordered_map(dict word2index):
    cdef unordered_map[string, unsigned int] word2index_map
    for key, value in word2index.items():
        word2index_map[key.encode("utf-8")] = value
    return word2index_map

cdef class Vocab:
    """
    Vocab
    -----

    Holds a mapping between words and indices.

    """
    cdef CVocab vocabinternal
    def __cinit__(self, list words = [], bint unknown_word = True):
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
