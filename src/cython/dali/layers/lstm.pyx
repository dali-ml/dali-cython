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


cdef vector[CLSTMState] ensure_state_list(object states):
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
            return dtype_dali_to_np(self.o.input_layer.b.dtype())

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