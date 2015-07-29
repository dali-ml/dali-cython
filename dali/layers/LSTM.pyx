from cython.operator cimport dereference as deref

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
        vector[CMat[T]] Wcells_to_inputs
        vector[CMat[T]] Wcells_to_forgets
        CStackedInputLayer[T] input_layer
        vector[CStackedInputLayer[T]] forget_layers
        CStackedInputLayer[T] output_layer
        CStackedInputLayer[T] cell_layer

        bint memory_feeds_gates
        bint backprop_through_gates
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
        vector[CLSTM[T]] cells
        bint shortcut
        bint memory_feeds_gates
        vector[CLSTMState[T]] activate(vector[CLSTMState[T]] previous_state, CMat[T] inpt, T drop_prob) except +
        vector[CLSTMState[T]] activate(vector[CLSTMState[T]] previous_state, vector[CMat[T]] inpt, T drop_prob) except +
        vector[CMat[T]] parameters() const
        CStackedLSTM();
        CStackedLSTM(const int& input_size, const vector[int]& hidden_sizes, bint shortcut, bint memory_feeds_gates)
        CStackedLSTM(const vector[int]& input_size, const vector[int]& hidden_sizes, bint shortcut, bint memory_feeds_gates)
        vector[CLSTMState[T]] initial_states() const
        CStackedLSTM[T] shallow_copy() const



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

        def __set__(LSTMState self, Mat value):
            self.lstmstateinternal.memory = value.matinternal


    property hidden:
        def __get__(self):
            return WrapMat(self.lstmstateinternal.hidden)

        def __set__(LSTMState self, Mat value):
            self.lstmstateinternal.hidden = value.matinternal

    def __setstate__(LSTM self, state):
        self.memory = state["memory"]
        self.hidden = state["hidden"]

    def __getstate__(self):
        return {
            "memory" : self.memory,
            "hidden" : self.hidden
        }

    def __reduce__(self):
        return (
            self.__class__,
            (), self.__getstate__(),
        )



cdef inline LSTMState WrapLSTMState(const CLSTMState[dtype]& internal):
    cdef LSTMState l = LSTMState()
    l.lstmstateinternal = internal
    return l

cdef inline list WrapLSTMStates(const vector[CLSTMState[dtype]]& lstm_states):
    ret = []
    cdef vector[CLSTMState[dtype]].const_iterator it = lstm_states.const_begin()

    while it != lstm_states.const_end():
        ret.append(WrapLSTMState(deref(it)))
        it += 1

    return ret

cdef inline vector[CLSTMState[dtype]] list_lstmstate_to_vector_lstmstate(list lstmstates):
    cdef vector[CLSTMState[dtype]] lstmstates_vec
    lstmstates_vec.reserve(len(lstmstates))
    for lstmstate in lstmstates:
        lstmstates_vec.push_back((<LSTMState>lstmstate).lstmstateinternal)
    return lstmstates_vec

cdef class LSTM:
    cdef CLSTM[dtype] layerinternal

    property Wcells_to_inputs:
        def __get__(LSTM self):
            return [WrapMat(m) for m in self.layerinternal.Wcells_to_inputs]

    property Wcells_to_forgets:
        def __get__(LSTM self):
            return [WrapMat(m) for m in self.layerinternal.Wcells_to_inputs]

    property input_layer:
        def __get__(LSTM self):
            return WrapStackedInputLayer(self.layerinternal.input_layer)

    property forget_layers:
        def __get__(LSTM self):
            return [WrapStackedInputLayer(l) for l in self.layerinternal.forget_layers]

    property output_layer:
        def __get__(LSTM self):
            return WrapStackedInputLayer(self.layerinternal.output_layer)

    property cell_layer:
        def __get__(LSTM self):
            return WrapStackedInputLayer(self.layerinternal.cell_layer)

    property input_size:
        def __get__(LSTM self):
            assert len(self.layerinternal.input_sizes) == 1
            return self.layerinternal.input_sizes[0]

    property input_sizes:
        def __get__(LSTM self):
            return self.layerinternal.input_sizes

    property hidden_size:
        def __get__(LSTM self):
            return self.layerinternal.hidden_size

    property num_children:
        def __get__(LSTM self):
            return self.layerinternal.num_children

    property memory_feeds_gates:
        def __get__(LSTM self):
            return self.layerinternal.memory_feeds_gates

    property backprop_through_gates:
        def __get__(LSTM self):
            return self.layerinternal.backprop_through_gates

        def __set__(LSTM self, bint should_backprop_through_gates):
            self.layerinternal.backprop_through_gates = should_backprop_through_gates

    def __cinit__(LSTM self, input_sizes, hidden_size, num_children=1, memory_feeds_gates=False):
        if type(input_sizes) == list:
            self.layerinternal = CLSTM[dtype](<vector[int]> input_sizes, <int> hidden_size, <int> num_children, <bint> memory_feeds_gates)
        elif type(input_sizes) == int:
            self.layerinternal = CLSTM[dtype](<int> input_sizes, <int> hidden_size, <int> num_children, <bint> memory_feeds_gates)
        else:
            raise ValueError("LSTM input_sizes must be a list or int, not " + type(input_sizes))

    def __call__(LSTM self, *args, **kwargs):
        return self.activate(*args, **kwargs)

    def activate(LSTM self, inpt, previous_states):
        cdef vector[CMat[dtype]]       inpt_vector
        cdef vector[CLSTMState[dtype]] previous_states_vector
        if type(inpt) != list:
            inpt = [inpt]
        inpt_vector = list_mat_to_vector_mat(inpt)

        if type(previous_states) != list:
            previous_states = [previous_states]
        previous_states_vector = list_lstmstate_to_vector_lstmstate(previous_states)

        return WrapLSTMState(
            self.layerinternal.activate_many_inputs(inpt_vector, previous_states_vector)
        )

    def activate_sequence(LSTM self, inputs, initial_state=None):
        cdef vector[CMat[dtype]]       inputs_vector

        if initial_state is None:
            initial_state = self.initial_state()
        inputs_vector = list_mat_to_vector_mat(inputs)
        return WrapLSTMState(
            self.layerinternal.activate_sequence((<LSTMState>initial_state).lstmstateinternal, inputs_vector)
        )

    def initial_states(LSTM self):
        return WrapLSTMState(
            self.layerinternal.initial_states()
        )

    def shallow_copy(LSTM self):
        cdef LSTM copy = LSTM(0,0)
        copy.layerinternal = self.layerinternal.shallow_copy()
        return copy

    def parameters(LSTM self):
        params = []
        cdef vector[CMat[dtype]] params_mat = self.layerinternal.parameters()
        for param in params_mat:
            mat = Mat(0,0)
            mat.matinternal = param
            params.append(mat)
        return params

    def __setstate__(LSTM self, state):
        for param, saved_param in zip(self.parameters(), state["parameters"]):
            param.w = saved_param.w
        self.backprop_through_gates = state["backprop_through_gates"]

    def __getstate__(self):
        return {
            "parameters" : self.parameters(),
            "backprop_through_gates" : self.backprop_through_gates
        }

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.input_sizes,
                self.hidden_size,
                self.num_children,
                self.memory_feeds_gates
            ), self.__getstate__(),
        )

    def __str__(LSTM self):
        child_string = '' if self.num_children == 1 else ', num_children=%d' % (self.num_children,)
        return "<LSTM inputs=%s, hidden_size=%d%s>" % (self.input_sizes, self.hidden_size, child_string)

    def __repr__(LSTM self):
        return str(self)

cdef inline LSTM WrapLSTM(const CLSTM[dtype]& internal):
    cdef LSTM output = LSTM(0,0)
    output.layerinternal = internal
    return output

cdef class StackedLSTM:
    cdef CStackedLSTM[dtype] layerinternal

    property cells:
        def __get__(self):
            return [WrapLSTM(l) for l in self.layerinternal.cells]

        def __set__(StackedLSTM self, list cells):
            cdef vector[CLSTM[dtype]] newcells
            for cell in cells:
                newcells.push_back((<LSTM>cell).layerinternal)

            self.layerinternal.cells = newcells


    def __cinit__(self, input_sizes, hidden_sizes, shortcut=False, memory_feeds_gates=False):
        if type(input_sizes) == list:
            self.layerinternal = CStackedLSTM[dtype](<vector[int]> input_sizes, <vector[int]> hidden_sizes, <bint> shortcut, <bint> memory_feeds_gates)
        elif type(input_sizes) == int:
            self.layerinternal = CStackedLSTM[dtype](<int> input_sizes, <vector[int]> hidden_sizes, <bint> shortcut, <bint> memory_feeds_gates)
        else:
            raise ValueError("list of int required for input_sizes for StackedLSTM constructor not " + type(input_sizes))

    def activate(self, inputs, hiddens, drop_prob = 0.0):
        cdef vector[CMat[dtype]] inputs_vector
        cdef vector[CLSTMState[dtype]] hiddens_vector
        hiddens_vector = list_lstmstate_to_vector_lstmstate(hiddens)
        if type(inputs) == list:
            inputs_vector = list_mat_to_vector_mat(inputs)
            return WrapLSTMStates(
                self.layerinternal.activate(hiddens_vector, inputs_vector, <dtype> drop_prob)
            )
        elif type(inputs) == Mat:
            return WrapLSTMStates(
                self.layerinternal.activate(hiddens_vector, (<Mat>inputs).matinternal, <dtype> drop_prob)
            )
        else:
            raise Exception("list or Mat expected for StackedLSTM activate not " + type(inputs))

    def shallow_copy(self):
        cdef StackedLSTM copy = StackedLSTM(0,0)
        copy.layerinternal = self.layerinternal.shallow_copy()
        return copy

    def parameters(self):
        params = []
        cdef vector[CMat[dtype]] params_mat = self.layerinternal.parameters()
        for param in params_mat:
            params.append(WrapMat(param))
        return params

    def initial_states(StackedLSTM self):
        return WrapLSTMStates(
            self.layerinternal.initial_states()
        )

    def __setstate__(LSTM self, state):
        self.cells = state["cells"]

    def __getstate__(self):
        return {
            "cells" : self.cells
        }

    def __reduce__(self):
        return (
            self.__class__,
            (
                [],
                [],
                False,
                False
            ), self.__getstate__(),
        )

    def __str__(StackedLSTM self):
        return "<StackedLSTM cells=%r>" % (self.cells)

    def __repr__(StackedLSTM self):
        return str(self)

