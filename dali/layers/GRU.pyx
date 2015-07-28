cdef extern from "dali/layers/GRU.h":
    cdef cppclass CGRU "GRU" [T]:
        CStackedInputLayer[T] reset_layer
        CStackedInputLayer[T] memory_interpolation_layer
        CStackedInputLayer[T] memory_to_memory_layer
        int input_size
        int hidden_size

        CGRU()
        CGRU(int _input_size, int _hidden_size)
        CGRU(const CGRU[T]&, bint copy_w, bint copy_dw)

        CGRU[T] shallow_copy() const
        CMat[T] activate(CMat[T] input_vector, CMat[T] previous_state) except +
        CMat[T] activate_sequence(
            const vector[CMat[T]]& input_sequence) except +
        CMat[T] activate_sequence(
            const vector[CMat[T]]& input_sequence,
            CMat[T] state) except +

        vector[CMat[T]] parameters() const
        CMat[T] initial_states() const

cdef class GRU:
    cdef CGRU[dtype] layerinternal

    def __cinit__(GRU self, int input_size, int hidden_size):
        self.layerinternal = CGRU[dtype](input_size, hidden_size)

    property reset_layer:
        def __get__(GRU self):
            cdef StackedInputLayer reset_layer = StackedInputLayer([],0)
            reset_layer.layerinternal = self.layerinternal.reset_layer
            return reset_layer

    property input_size:
        def __get__(GRU self):
            return self.layerinternal.input_size

    property hidden_size:
        def __get__(GRU self):
            return self.layerinternal.hidden_size

    def activate(GRU self, Mat input_vector, Mat previous_state):
        """
        Activate
        --------

        Inputs
        ------

        Mat input_vector
        Mat previous_state

        Outputs
        -------

        Mat next_state
        """
        return WrapMat(
            self.layerinternal.activate(
                input_vector.matinternal,
                previous_state.matinternal
            )
        )

    def initial_states(GRU self):
        return WrapMat(self.layerinternal.initial_states())

    def parameters(GRU self):
        params = []
        cdef vector[CMat[dtype]] params_mat = self.layerinternal.parameters()
        for param in params_mat:
            params.append(WrapMat(param))
        return params

    def activate_sequence(GRU self, list input_sequence, initial_state = None):
        cdef vector[CMat[dtype]] mats = list_mat_to_vector_mat(input_sequence)
        if initial_state is None:
            return WrapMat(self.layerinternal.activate_sequence(mats))
        else:
            return WrapMat(self.layerinternal.activate_sequence(
                mats,
                (<Mat> initial_state).matinternal
            ))
