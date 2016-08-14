import numpy as np

cdef class Layer:
    def __cinit__(Layer self,
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
            self.o = CLayer()
        else:
            self.o = CLayer(input_size, hidden_size, dali_dtype, device.o)

    property W:
        def __get__(Layer self):
            return Tensor.wrapc(self.o.W)
        def __set__(Layer self, Tensor W):
            self.o.W = W.o

    property input_size:
        def __get__(Layer self):
            return self.o.input_size

    property hidden_size:
        def __get__(Layer self):
            return self.o.hidden_size

    property dtype:
        def __get__(Layer self):
            return dtype_dali_to_np(self.o.dtype)

    property b:
        def __get__(Layer self):
            return Tensor.wrapc(self.o.b)
        def __set__(Layer self, Tensor b):
            self.o.b = b.o

    @staticmethod
    cdef Layer wrapc(CLayer o):
        ret = Layer(0,0)
        ret.o = o
        return ret

    def activate(Layer self, Tensor x):
        return Tensor.wrapc(self.o.activate(x.o))

    def parameters(Layer self):
        return ctensors_to_list(self.o.parameters())

    def __setstate__(Layer self, state):
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

cdef class StackedInputLayer:
    def __cinit__(Layer self,
                  vector[int] input_sizes,
                  int hidden_size,
                  dtype=None,
                  preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        cdef c_np.NPY_TYPES c_np_dtype = c_np.NPY_FLOAT32
        if dtype is not None:
            c_np_dtype = c_np.dtype(dtype).num
        cdef DType dali_dtype = dtype_c_np_to_dali(c_np_dtype)
        if len(input_sizes) == 0 and hidden_size == 0:
            self.o = CStackedInputLayer()
        else:
            self.o = CStackedInputLayer(input_sizes, hidden_size, dali_dtype, device.o)

    def activate(StackedInputLayer self, tensors):
        cdef vector[CTensor] tensors_c = ensure_tensor_list(tensors)
        return Tensor.wrapc(self.o.activate(tensors_c))

    @staticmethod
    cdef StackedInputLayer wrapc(CStackedInputLayer o):
        ret = StackedInputLayer([],0)
        ret.o = o
        return ret

    def parameters(Layer self):
        return ctensors_to_list(self.o.parameters())

    property dtype:
        def __get__(StackedInputLayer self):
            return dtype_dali_to_np(self.o.dtype)

    property input_sizes:
        def __get__(StackedInputLayer self):
            return self.o.get_input_sizes()

    property hidden_size:
        def __get__(StackedInputLayer self):
            return self.o.hidden_size

    property b:
        def __get__(StackedInputLayer self):
            return Tensor.wrapc(self.o.b)
        def __set__(StackedInputLayer self, Tensor b):
            self.o.b = b.o

    property tensors:
        def __get__(StackedInputLayer self):
            return ctensors_to_list(self.o.tensors)

        def __set__(StackedInputLayer self, tensors):
            cdef vector[CTensor] tensors_c = ensure_tensor_list(tensors)
            self.o.tensors = tensors_c

    def __setstate__(StackedInputLayer self, state):
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
                self.input_sizes,
                self.hidden_size,
                self.dtype,
                Device.wrapc(self.o.device)
            ), self.__getstate__(),
        )
