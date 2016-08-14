import numpy as np

cdef class LSTMState:
    def __cinit__(LSTMState self, memory, hidden):
        if memory is None and hidden is None:
            self.o = CLSTMState()
        else:
            self.o = CLSTMState((<Tensor>memory).o, (<Tensor>hidden).o)

    property memory:
        def __get__(LSTMState self):
            return Tensor.wrapc(self.o.memory)

        def __set__(LSTMState self, Tensor memory):
            self.o.memory = memory.o

    property hidden:
        def __get__(LSTMState self):
            return Tensor.wrapc(self.o.hidden)

        def __set__(LSTMState self, Tensor hidden):
            self.o.hidden = hidden.o

    @staticmethod
    cdef LSTMState wrapc(CLSTMState o):
        ret = LSTMState(None, None)
        ret.o = o
        return ret

    def __setstate__(LSTMState self, state):
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
            (None, None), self.__getstate__(),
        )


cdef vector[CLSTMState] ensure_state_list(object states) except +:
    cdef vector[CLSTMState] states_c

    got_list_of_states = (
        isinstance(states, (tuple, list)) and
        all([type(t) == LSTMState for t in states])
    )

    if not got_list_of_states:
        raise ValueError("expected a list or a tuple of LSTMState.")

    for state in states:
        states_c.emplace_back((<LSTMState>state).o)

    return states_c

cdef list clstm_states_to_list(const vector[CLSTMState]& cstates) except +:
    out = []
    for i in range(cstates.size()):
        out.append(LSTMState.wrapc(cstates[i]))
    return out

cdef class LSTM:
    def __cinit__(LSTM self,
                  input_size,
                  int hidden_size,
                  int num_children=1,
                  bint memory_feeds_gates=False,
                  dtype=None,
                  preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        cdef c_np.NPY_TYPES c_np_dtype = c_np.NPY_FLOAT32
        if dtype is not None:
            c_np_dtype = c_np.dtype(dtype).num
        cdef DType dali_dtype = dtype_c_np_to_dali(c_np_dtype)
        cdef vector[int] input_sizes
        if input_size == 0 and hidden_size == 0:
            self.o = CLSTM()
        else:
            if isinstance(input_size, int):
                input_sizes.emplace_back(<int>input_size)
            else:
                input_sizes = input_size
            self.o = CLSTM(input_sizes, hidden_size, num_children, memory_feeds_gates, dali_dtype, device.o)

    property Wco:
        def __get__(LSTM self):
            return Tensor.wrapc(self.o.Wco)

        def __set__(LSTM self, Tensor Wco):
            self.o.Wco = Wco.o

    property Wcells_to_inputs:
        def __get__(LSTM self):
            return ctensors_to_list(self.o.Wcells_to_inputs)

    property Wcells_to_forgets:
        def __get__(LSTM self):
            return ctensors_to_list(self.o.Wcells_to_forgets)

    property input_size:
        def __get__(LSTM self):
            if len(self.o.input_sizes) > 1:
                raise ValueError(
                    "LSTM with multiple input_sizes does not "
                    "does not have a single input_size (use "
                    "`input_sizes` instead)."
                )
            return self.o.input_sizes[0]

    property input_sizes:
        def __get__(LSTM self):
            return self.o.input_sizes

    property hidden_size:
        def __get__(LSTM self):
            return self.o.hidden_size

    property dtype:
        def __get__(LSTM self):
            return dtype_dali_to_np(self.o.dtype)

    property backprop_through_gates:
        def __get__(LSTM self):
            return self.o.backprop_through_gates

        def __set__(LSTM self, bint backprop_through_gates):
            self.o.backprop_through_gates = backprop_through_gates

    property memory_feeds_gates:
        def __get__(LSTM self):
            return self.o.memory_feeds_gates

    property num_children:
        def __get__(LSTM self):
            return self.o.num_children

    property input_layer:
        def __get__(LSTM self):
            return StackedInputLayer.wrapc(self.o.input_layer)

    property cell_layer:
        def __get__(LSTM self):
            return StackedInputLayer.wrapc(self.o.cell_layer)

    property output_layer:
        def __get__(LSTM self):
            return StackedInputLayer.wrapc(self.o.output_layer)

    property forget_layers:
        def __get__(LSTM self):
            forget_layers = []
            for layer in self.o.forget_layers:
                forget_layers.append(StackedInputLayer.wrapc(layer))
            return forget_layers

    def initial_states(LSTM self):
        return LSTMState.wrapc(self.o.initial_states())

    @staticmethod
    cdef LSTM wrapc(CLSTM o):
        ret = LSTM(0,0)
        ret.o = o
        return ret

    def activate(LSTM self, x, state):
        cdef bint state_is_list = isinstance(state, (tuple, list))
        cdef bint x_is_list = isinstance(x, (tuple, list))
        cdef vector[CTensor] xs
        cdef vector[CLSTMState] states

        if state_is_list or x_is_list:
            states = ensure_state_list(state)
            if x_is_list:
                xs = ensure_tensor_list(x)
            else:
                xs = ensure_tensor_list([x])
            return LSTMState.wrapc(self.o.activate(xs, states))
        else:
            return LSTMState.wrapc(self.o.activate((<Tensor>x).o, (<LSTMState>state).o))

    def parameters(LSTM self):
        return ctensors_to_list(self.o.parameters())

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
                self.memory_feeds_gates,
                self.dtype,
                Device.wrapc(self.o.device)
            ), self.__getstate__(),
        )


cdef vector[CLSTM] ensure_lstm_list(object cells) except +:
    cdef vector[CLSTM] cells_c

    got_list_of_cells = (
        isinstance(cells, (tuple, list)) and
        all([type(t) == LSTM for t in cells])
    )

    if not got_list_of_cells:
        raise ValueError("expected a list or a tuple of LSTM.")

    for cell in cells:
        cells_c.emplace_back((<LSTM>cell).o)

    return cells_c


cdef class StackedLSTM:
    def __cinit__(StackedLSTM self,
                  input_size,
                  vector[int] hidden_sizes,
                  bint shortcut=False,
                  bint memory_feeds_gates=False,
                  dtype=None,
                  preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        cdef c_np.NPY_TYPES c_np_dtype = c_np.NPY_FLOAT32
        if dtype is not None:
            c_np_dtype = c_np.dtype(dtype).num
        cdef DType dali_dtype = dtype_c_np_to_dali(c_np_dtype)
        cdef vector[int] input_sizes
        if len(hidden_sizes) == 0:
            self.o = CStackedLSTM()
        else:
            if isinstance(input_size, int):
                input_sizes.emplace_back(<int>input_size)
            else:
                input_sizes = input_size
            self.o = CStackedLSTM(input_sizes, hidden_sizes, shortcut, memory_feeds_gates, dali_dtype, device.o)

    property input_size:
        def __get__(StackedLSTM self):
            cdef vector[int] sizes = self.o.input_sizes()
            if len(sizes.size()) > 1:
                raise ValueError(
                    "StackedLSTM with multiple input_sizes does "
                    "not does not have a single input_size (use "
                    "`input_sizes` instead)."
                )
            return sizes[0]

    property input_sizes:
        def __get__(StackedLSTM self):
            return self.o.input_sizes()

    property hidden_sizes:
        def __get__(StackedLSTM self):
            return self.o.hidden_sizes()

    property cells:
        def __get__(StackedLSTM self):
            out = []
            cdef int i
            for i in range(self.o.cells.size()):
                out.append(LSTM.wrapc(self.o.cells[i]))
            return out

        def __set__(StackedLSTM self, cells):
            cdef vector[CLSTM] c_cells = ensure_lstm_list(cells)
            cdef DType dali_dtype = self.o.dtype
            cdef int i = 0
            for i in range(c_cells.size()):
                if c_cells[i].dtype != dali_dtype:
                    if i > 0:
                        raise ValueError(
                            ("all cells must have the same "
                             "dtype (got %r vs. %r)") % (
                                dtype_dali_to_np(c_cells[i].dtype),
                                dtype_dali_to_np(dali_dtype)
                            )
                        )
                    else:
                        dali_dtype = c_cells[i].dtype
            self.o.dtype = dali_dtype
            self.o.cells = c_cells

    property dtype:
        def __get__(StackedLSTM self):
            return dtype_dali_to_np(self.o.dtype)

    property shortcut:
        def __get__(StackedLSTM self):
            return self.o.shortcut

    def initial_states(StackedLSTM self):
        return clstm_states_to_list(self.o.initial_states())

    @staticmethod
    cdef StackedLSTM wrapc(CStackedLSTM o):
        ret = StackedLSTM(0,[])
        ret.o = o
        return ret

    def activate(StackedLSTM self, x, state, double drop_prob=0.0):
        cdef bint x_is_list = isinstance(x, (tuple, list))
        cdef vector[CTensor] xs
        cdef vector[CLSTMState] states = ensure_state_list(state)

        if x_is_list:
            xs = ensure_tensor_list(x)
            return clstm_states_to_list(self.o.activate(xs, states, drop_prob))
        else:
            return clstm_states_to_list(self.o.activate((<Tensor>x).o, states, drop_prob))

    def parameters(StackedLSTM self):
        return ctensors_to_list(self.o.parameters())

    def __setstate__(LSTM self, state):
        self.cells = state["cells"]
        self.o.shortcut = state["shortcut"]
        self.o.device = (<Device>state["device"]).o
        self.o.dtype = dtype_np_to_dali(state["dtype"])

    def __getstate__(self):
        return {
            "cells" : self.cells,
            "shortcut": self.shortcut,
            "dtype": self.dtype,
            "device": Device.wrapc(self.o.device)
        }

    def __reduce__(self):
        return (
            self.__class__,
            (
                0,
                []
            ), self.__getstate__(),
        )

