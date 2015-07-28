cdef extern from "dali/layers/LSTM.h":
    cdef cppclass CLSTMState "LSTMState" [T]:
        CMat[T] memory
        CMat[T] hidden
        CLSTMState()
        CLSTMState(CMat[T] memory, CMat[T] hidden)
        @staticmethod
        vector[CMat[T]] hiddens(const vector[CMat[T]]&)
        @staticmethod
        vector[CMat[T]] memories(const vector[CMat[T]]&)

    cdef cppclass CLSTM "LSTM" [T]:
        int hidden_size
        int num_children
        vector[int] input_sizes
        bint memory_feeds_gates
        bint backprop_though_gates
        CLSTM()
        CLSTM(int input_size, int hidden_size, bint memory_feeds_gates)
        CLSTM(int input_size, int hidden_size, int num_children, bint memory_feeds_gates)
        CLSTM(vector[int] input_sizes, int hidden_size, int num_children, bint memory_feeds_gates)
        CLSTM(const CLSTM& other, bint copy_w, bint copy_dw)
        vector[CMat[T]] parameters() const
        @staticmethod
        vector[CLSTMState[T]] initial_states(const vector[int]& hidden_sizes)
        CLSTMState[T] initial_states() const

        CLSTMState[T] activate(CMat[T] input_vector, CLSTMState[T] previous_state)  except +
        CLSTMState[T] activate_children "activate"(CMat[T] input_vector, vector[CLSTMState[T]] previous_children_states)  except +
        CLSTMState[T] activate_many_inputs "activate"(vector[CMat[T]] input_vectors, vector[CLSTMState[T]] previous_children_states) except +
        CLSTMState[T] activate_shortcut "activate"(CMat[T] input_vector, CMat[T] shortcut_vector, CLSTMState[T] previous_children_state) except +
        CLSTM[T] shallow_copy() const
        CLSTMState[T] activate_sequence(CLSTMState[T], const vector[CMat[T]]& input_vectors) except +

    cdef cppclass CStackedLSTM "StackedLSTM" [T]:
        pass


cdef class LSTMState:
    cdef CLSTMState[dtype] lstmstateinternal

    def __cinit__(self, memory=None, hidden=None):
        self.lstmstateinternal = CLSTMState[dtype]()
        if memory is not None:
            self.lstmstateinternal.memory = (<Mat>memory).matinternal
        if hidden is not None:
            self.lstmstateinternal.hidden = (<Mat>memory).matinternal

    property memory:
        def __get__(self):
            return WrapMat(self.lstmstateinternal.memory)

    property hidden:
        def __get__(self):
            return WrapMat(self.lstmstateinternal.hidden)

cdef inline LSTMState WrapLSTMState(const CLSTMState[dtype]& internal):
    l = LSTMState()
    l.lstmstateinternal = internal
    return l

cdef inline vector[CLSTMState[dtype]] list_lstmstate_to_vector_lstmstate(list lstmstates):
    cdef vector[CLSTMState[dtype]] lstmstates_vec
    lstmstates_vec.reserve(len(lstmstates))
    for lstmstate in lstmstates:
        lstmstates_vec.push_back((<LSTMState>lstmstate).lstmstateinternal)
    return lstmstates_vec

cdef class LSTM:
    cdef CLSTM[dtype] lstminternal

    property input_size:
        def __get__(self):
            assert len(self.lstminternal.input_sizes) == 1
            return self.lstminternal.input_sizes[0]

    property input_sizes:
        def __get__(self):
            return self.lstminternal.input_sizes

    property hidden_size:
        def __get__(self):
            return self.lstminternal.hidden_size

    property num_children:
        def __get__(self):
            return self.lstminternal.num_children

    def __cinit__(self, input_sizes, hidden_size, num_children=1, memory_feeds_gates=False):
        if type(input_sizes) == list:
            self.lstminternal = CLSTM[dtype](<vector[int]> input_sizes, <int> hidden_size, <int> num_children, <bint> memory_feeds_gates)
        elif type(input_sizes) == int:
            self.lstminternal = CLSTM[dtype](<int> input_sizes, <int> hidden_size, <int> num_children, <bint> memory_feeds_gates)
        else:
            raise ValueError("LSTM input_sizes must be a list or int, not " + type(input_sizes))

    def __call__(self, *args, **kwargs):
        return self.activate(*args, **kwargs)

    def activate(self, inpt, previous_states):
        cdef vector[CMat[dtype]]       inpt_vector
        cdef vector[CLSTMState[dtype]] previous_states_vector
        if type(inpt) != list:
            inpt = [inpt]
        inpt_vector = list_mat_to_vector_mat(inpt)

        if type(previous_states) != list:
            previous_states = [previous_states]
        previous_states_vector = list_lstmstate_to_vector_lstmstate(previous_states)

        return WrapLSTMState(
            self.lstminternal.activate_many_inputs(inpt_vector, previous_states_vector)
        )

    def activate_sequence(self, inputs, initial_state=None):
        cdef vector[CMat[dtype]]       inputs_vector

        if initial_state is None:
            initial_state = self.initial_state()
        inputs_vector = list_mat_to_vector_mat(inputs)
        return WrapLSTMState(
            self.lstminternal.activate_sequence((<LSTMState>initial_state).lstmstateinternal, inputs_vector)
        )

    def initial_states(self):
        return WrapLSTMState(
            self.lstminternal.initial_states()
        )

    def shallow_copy(self):
        cdef LSTM copy = LSTM(0,0)
        copy.lstminternal = self.lstminternal.shallow_copy()
        return copy

    def parameters(self):
        params = []
        cdef vector[CMat[dtype]] params_mat = self.lstminternal.parameters()
        for param in params_mat:
            mat = Mat(0,0)
            mat.matinternal = param
            params.append(mat)
        return params
