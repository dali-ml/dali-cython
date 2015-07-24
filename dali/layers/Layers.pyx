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
        CRNN[T] shallow_copy() const
        vector[CMat[T]] parameters() const

    cdef cppclass CLayer "Layer" [T]:
        int hidden_size
        int input_size
        CMat[T] W
        vector[CMat[T]] parameters() const
        # constructors
        CLayer()
        CLayer(int input_size, int hidden_size)
        CLayer(const CLayer& other, bint copy_w, bint copy_dw)

        CMat[T] activate(CMat[T]) const
        CLayer[T] shallow_copy() const

cdef class Layer:
    cdef CLayer[dtype] layerinternal

    property input_size:
        def __get__(self):
            return self.layerinternal.input_size

    property hidden_size:
        def __get__(self):
            return self.layerinternal.hidden_size

    def __cinit__(self, int input_size, int hidden_size):
        self.layerinternal = CLayer[dtype](input_size, hidden_size)

    def activate(self, Mat input_vector):
        cdef Mat output = Mat(0,0)
        output.matinternal = self.layerinternal.activate(input_vector.matinternal)
        return output

    def shallow_copy(self):
        cdef Layer copy = Layer(0,0)
        copy.layerinternal = self.layerinternal.shallow_copy()
        return copy

    def parameters(self):
        params = []
        cdef vector[CMat[dtype]] params_mat = self.layerinternal.parameters()
        for param in params_mat:
            mat = Mat(0,0)
            mat.matinternal = param
            params.append(mat)
        return params

cdef class RNN:
    cdef CRNN[dtype] layerinternal

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
        self.layerinternal = CRNN[dtype](input_size, hidden_size, out_size)

    def activate(self, Mat input_vector, Mat prev_hidden):
        cdef Mat output = Mat(0,0)
        output.matinternal = self.layerinternal.activate(input_vector.matinternal, prev_hidden.matinternal)
        return output

    def shallow_copy(self):
        cdef RNN copy = RNN(0,0)
        copy.layerinternal = self.layerinternal.shallow_copy()
        return copy

    def parameters(self):
        params = []
        cdef vector[CMat[dtype]] params_mat = self.layerinternal.parameters()
        for param in params_mat:
            mat = Mat(0,0)
            mat.matinternal = param
            params.append(mat)
        return params
