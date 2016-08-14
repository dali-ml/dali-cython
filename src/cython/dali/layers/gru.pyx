import numpy as np

cdef class GRU:
    def __cinit__(GRU self,
                  int input_size,
                  int hidden_size,
                  dtype=None,
                  preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        cdef c_np.NPY_TYPES c_np_dtype = c_np.NPY_FLOAT32
        if dtype is not None:
            c_np_dtype = c_np.dtype(dtype).num
        cdef DType dali_dtype = dtype_c_np_to_dali(c_np_dtype)
        if input_size == 0 and hidden_size == 0:
            self.o = CGRU()
        else:
            self.o = CGRU(input_size, hidden_size, dali_dtype, device.o)

    property input_size:
        def __get__(GRU self):
            return self.o.input_size

    property hidden_size:
        def __get__(GRU self):
            return self.o.hidden_size

    property dtype:
        def __get__(GRU self):
            return dtype_dali_to_np(self.o.dtype)

    property reset_layer:
        def __get__(GRU self):
            return StackedInputLayer.wrapc(self.o.reset_layer)

    property memory_interpolation_layer:
        def __get__(GRU self):
            return StackedInputLayer.wrapc(self.o.memory_interpolation_layer)

    property memory_to_memory_layer:
        def __get__(GRU self):
            return StackedInputLayer.wrapc(self.o.memory_to_memory_layer)

    def initial_states(GRU self):
        return Tensor.wrapc(self.o.initial_states())

    @staticmethod
    cdef GRU wrapc(CGRU o):
        ret = GRU(0,0)
        ret.o = o
        return ret

    def activate(GRU self, Tensor x, Tensor state):
        return Tensor.wrapc(self.o.activate(x.o, state.o))

    def parameters(GRU self):
        return ctensors_to_list(self.o.parameters())

    def __setstate__(GRU self, state):
        for param, saved_param in zip(self.parameters(), state["parameters"]):
            param.w = saved_param.w

    def __getstate__(self):
        return {
            "parameters" : self.parameters()
        }

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.input_size,
                self.hidden_size,
                self.dtype,
                Device.wrapc(self.o.device)
            ), self.__getstate__(),
        )
