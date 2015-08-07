

cdef extern from "dali/layers/GRU.h" nogil:
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
    cdef void* layerinternal
    cdef np.NPY_TYPES dtypeinternal

    def __cinit__(GRU self, int input_size, int hidden_size, dtype=np.float32):
        self.layerinternal = NULL
        self.dtypeinternal = np.NPY_NOTYPE
        self.dtypeinternal = np.dtype(dtype).num
        ensure_fdtype(self.dtypeinternal)

        if (self).dtypeinternal == np.NPY_FLOAT32:
            self.layerinternal = new CGRU[float](input_size, hidden_size)
        elif (self).dtypeinternal == np.NPY_FLOAT64:
            self.layerinternal = new CGRU[double](input_size, hidden_size)
        else:
            raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")


    def __dealloc__(GRU self):
        self.free_internal()

    cdef free_internal(GRU self):
        cdef CGRU[float]* ptr_internal_float
        cdef CGRU[double]* ptr_internal_double

        if self.layerinternal != NULL:
            if (self).dtypeinternal == np.NPY_FLOAT32:
                ptr_internal_float = (<CGRU[float]*>((<GRU>(self)).layerinternal))
                with nogil:
                    del ptr_internal_float
            elif (self).dtypeinternal == np.NPY_FLOAT64:
                ptr_internal_double = (<CGRU[double]*>((<GRU>(self)).layerinternal))
                with nogil:
                    del ptr_internal_double
            else:
                raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")

            self.layerinternal = NULL

    property dtype:
        def __get__(GRU self):
            return np.PyArray_DescrFromType(self.dtypeinternal)

    property input_size:
        def __get__(GRU self):
            if (self).dtypeinternal == np.NPY_FLOAT32:
                return (<CGRU[float]*>((<GRU>(self)).layerinternal))[0].input_size
            elif (self).dtypeinternal == np.NPY_FLOAT64:
                return (<CGRU[double]*>((<GRU>(self)).layerinternal))[0].input_size
            else:
                raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")

    property hidden_size:
        def __get__(GRU self):
            if (self).dtypeinternal == np.NPY_FLOAT32:
                return (<CGRU[float]*>((<GRU>(self)).layerinternal))[0].hidden_size
            elif (self).dtypeinternal == np.NPY_FLOAT64:
                return (<CGRU[double]*>((<GRU>(self)).layerinternal))[0].hidden_size
            else:
                raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")


    property reset_layer:
        def __get__(GRU self):
            if (self).dtypeinternal == np.NPY_FLOAT32:
                return WrapStackedLayer_float((<CGRU[float]*>((<GRU>(self)).layerinternal))[0].reset_layer)
            elif (self).dtypeinternal == np.NPY_FLOAT64:
                return WrapStackedLayer_double((<CGRU[double]*>((<GRU>(self)).layerinternal))[0].reset_layer)
            else:
                raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")

    property memory_interpolation_layer:
        def __get__(GRU self):
            if (self).dtypeinternal == np.NPY_FLOAT32:
                return WrapStackedLayer_float((<CGRU[float]*>((<GRU>(self)).layerinternal))[0].memory_interpolation_layer)
            elif (self).dtypeinternal == np.NPY_FLOAT64:
                return WrapStackedLayer_double((<CGRU[double]*>((<GRU>(self)).layerinternal))[0].memory_interpolation_layer)
            else:
                raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")

    property memory_to_memory_layer:
        def __get__(GRU self):
            if (self).dtypeinternal == np.NPY_FLOAT32:
                return WrapStackedLayer_float((<CGRU[float]*>((<GRU>(self)).layerinternal))[0].memory_to_memory_layer)
            elif (self).dtypeinternal == np.NPY_FLOAT64:
                return WrapStackedLayer_double((<CGRU[double]*>((<GRU>(self)).layerinternal))[0].memory_to_memory_layer)
            else:
                raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")


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
        assert(
            (self.dtypeinternal == input_vector.dtypeinternal) and
            (previous_state.dtypeinternal == self.dtypeinternal)
        ), "All arguments must be of the same type"

        cdef CMat[float] out_float
        cdef CMat[double] out_double


        if (self).dtypeinternal == np.NPY_FLOAT32:
            with nogil:
                out_float = (<CGRU[float]*>((<GRU>(self)).layerinternal))[0].activate((<CMat[float]*>((<Mat>(input_vector)).matinternal))[0], (<CMat[float]*>((<Mat>(previous_state)).matinternal))[0])
            return WrapMat_float(out_float)
        elif (self).dtypeinternal == np.NPY_FLOAT64:
            with nogil:
                out_double = (<CGRU[double]*>((<GRU>(self)).layerinternal))[0].activate((<CMat[double]*>((<Mat>(input_vector)).matinternal))[0], (<CMat[double]*>((<Mat>(previous_state)).matinternal))[0])
            return WrapMat_double(out_double)
        else:
            raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")


    def initial_states(GRU self):
        if (self).dtypeinternal == np.NPY_FLOAT32:
            return WrapMat_float((<CGRU[float]*>((<GRU>(self)).layerinternal))[0].initial_states())
        elif (self).dtypeinternal == np.NPY_FLOAT64:
            return WrapMat_double((<CGRU[double]*>((<GRU>(self)).layerinternal))[0].initial_states())
        else:
            raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")


    def parameters(GRU self):
        params = []
        cdef CMat[float]         param_float
        cdef vector[CMat[float]] param_vec_float
        
        cdef CMat[double]         param_double
        cdef vector[CMat[double]] param_vec_double
        

        if (self).dtypeinternal == np.NPY_FLOAT32:
            param_vec_float = (<CGRU[float]*>((<GRU>(self)).layerinternal))[0].parameters()
            for param_float in param_vec_float:
                params.append(WrapMat_float(param_float))
        elif (self).dtypeinternal == np.NPY_FLOAT64:
            param_vec_double = (<CGRU[double]*>((<GRU>(self)).layerinternal))[0].parameters()
            for param_double in param_vec_double:
                params.append(WrapMat_double(param_double))
        else:
            raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of np.float32, np.float64)")

        return params

    def __setstate__(GRU self, state):
        for param, saved_param in zip(self.parameters(), state["parameters"]):
            param.w = saved_param.w
            self.dtypeinternal = state["dtype"].num

    def __getstate__(self):
        return {
            "parameters" : self.parameters(),
            "dtype" : self.dtype
        }

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.input_size,
                self.hidden_size,
            ), self.__getstate__(),
        )

    def activate_sequence(GRU self, list input_sequence, initial_state = None):
        cdef vector[CMat[float]] c_input_sequence_float
        cdef CMat[float] out_float
        cdef vector[CMat[double]] c_input_sequence_double
        cdef CMat[double] out_double


        if initial_state is None:
            if len(input_sequence) == 0:
                raise ValueError('list cannot be empty')
            common_dtype = (<Mat>(input_sequence[0])).dtypeinternal
            for el in input_sequence:
                if (<Mat>el).dtypeinternal != common_dtype:
                    common_dtype = -1
                    break
            if common_dtype == -1:
                raise ValueError("All the arguments must be of the same type")
            if common_dtype == np.NPY_FLOAT32:
                c_input_sequence_float = mats_to_vec_float(input_sequence)
                if self.dtypeinternal != np.NPY_FLOAT32:
                    raise ValueError("Invalid dtype for input_sequence: " + str(input_sequence[0].dtype) + ", when GRU is " + str(self.dtype))
                with nogil:
                    out_float = (<CGRU[float]*>((<GRU>(self)).layerinternal))[0].activate_sequence(c_input_sequence_float)
                return WrapMat_float(out_float)
            elif common_dtype == np.NPY_FLOAT64:
                c_input_sequence_double = mats_to_vec_double(input_sequence)
                if self.dtypeinternal != np.NPY_FLOAT64:
                    raise ValueError("Invalid dtype for input_sequence: " + str(input_sequence[0].dtype) + ", when GRU is " + str(self.dtype))
                with nogil:
                    out_double = (<CGRU[double]*>((<GRU>(self)).layerinternal))[0].activate_sequence(c_input_sequence_double)
                return WrapMat_double(out_double)
            else:
                raise ValueError("Invalid dtype:" + str(input_sequence[0].dtype) + " (should be one of np.float32, np.float64)")

        else:
            if type(initial_state) is not Mat:
                raise ValueError("initial_state must be a Mat")
            if len(input_sequence) == 0:
                raise ValueError('list cannot be empty')
            common_dtype = (<Mat>(input_sequence[0])).dtypeinternal
            for el in input_sequence:
                if (<Mat>el).dtypeinternal != common_dtype:
                    common_dtype = -1
                    break
            if common_dtype == -1:
                raise ValueError("All the arguments must be of the same type")
            if common_dtype == np.NPY_FLOAT32:
                c_input_sequence_float = mats_to_vec_float(input_sequence)
                if self.dtypeinternal != np.NPY_FLOAT32:
                    raise ValueError("Invalid dtype for input_sequence: " + str(input_sequence[0].dtype) + ", when GRU is " + str(self.dtype))
                if (<Mat>initial_state).dtypeinternal != self.dtypeinternal:
                    raise ValueError("Invalid dtype for initial_state: " + str(initial_state.dtype) + ", when GRU is " + str(self.dtype))
                with nogil:
                    out_float = (<CGRU[float]*>((<GRU>(self)).layerinternal))[0].activate_sequence(c_input_sequence_float, (<CMat[float]*>((<Mat>(initial_state)).matinternal))[0])
                return WrapMat_float(out_float)
            elif common_dtype == np.NPY_FLOAT64:
                c_input_sequence_double = mats_to_vec_double(input_sequence)
                if self.dtypeinternal != np.NPY_FLOAT64:
                    raise ValueError("Invalid dtype for input_sequence: " + str(input_sequence[0].dtype) + ", when GRU is " + str(self.dtype))
                if (<Mat>initial_state).dtypeinternal != self.dtypeinternal:
                    raise ValueError("Invalid dtype for initial_state: " + str(initial_state.dtype) + ", when GRU is " + str(self.dtype))
                with nogil:
                    out_double = (<CGRU[double]*>((<GRU>(self)).layerinternal))[0].activate_sequence(c_input_sequence_double, (<CMat[double]*>((<Mat>(initial_state)).matinternal))[0])
                return WrapMat_double(out_double)
            else:
                raise ValueError("Invalid dtype:" + str(input_sequence[0].dtype) + " (should be one of np.float32, np.float64)")


    def __str__(GRU self):
        return "<GRU in=%d, hidden=%d>" % (self.input_size, self.hidden_size)

    def __repr__(GRU self):
        return str(self)

cdef inline GRU WrapGRU_float(const CGRU[float]& internal):
    cdef GRU output = GRU(0,0)
    output.free_internal()
    output.layerinternal = new CGRU[float](internal, False, False)
    output.dtypeinternal = np.NPY_FLOAT32
    return output
cdef inline GRU WrapGRU_double(const CGRU[double]& internal):
    cdef GRU output = GRU(0,0)
    output.free_internal()
    output.layerinternal = new CGRU[double](internal, False, False)
    output.dtypeinternal = np.NPY_FLOAT64
    return output

