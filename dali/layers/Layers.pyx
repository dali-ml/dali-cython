cdef extern from "dali/layers/Layers.h":
    cdef cppclass CRNN "RNN" [T]:
        int input_size
        int hidden_size
        int output_size
        CRNN()
        CRNN(int input_size, int hidden_size)
        CRNN(int input_size, int hidden_size, int output_size)
        CRNN(CRNN[T]&, bool, bool)
        CMat[T] activate(CMat[T] input_vector, CMat[T] prev_hidden) except +

cdef class RNN:
    cdef CRNN["double"] layerinternal

    property input_size:
        def __get__(self):
            return self.layerinternal.input_size

    property hidden_size:
        def __get__(self):
            return self.layerinternal.hidden_size

    property output_size:
        def __get__(self):
            return self.layerinternal.output_size

    def __cinit__(self, int input_size, int hidden_size, output_size = None):
        if output_size is None:
            output_size = hidden_size
        assert(input_size > -1 and hidden_size > -1 and output_size > -1), "Only positive dimensions may be used."
        cdef int out_size = output_size
        self.layerinternal = CRNN["double"](input_size, hidden_size, out_size)

    def activate(self, Mat input_vector, Mat prev_hidden):
        cdef Mat output = Mat(0,0)
        output.matinternal = self.layerinternal.activate(input_vector.matinternal, prev_hidden.matinternal)
        return output
