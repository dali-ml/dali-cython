import numpy as np


cdef class Array:
    def __cinit__(Array self, vector[int] shape, dtype=np.float32, preferred_device=None):
        cdef Device device
        device = ensure_device(preferred_device)

        self.o = CArray(shape, dtype_np_to_dali(dtype), device.o)

    @staticmethod
    cdef Array wrapc(CArray o):
        ret = Array([])
        ret.o = o
        return ret

    property dtype:
        def __get__(Array self):
            return dtype_dali_to_np(self.o.dtype())

    property shape:
        def __get__(Array self):
            return self.o.shape()

    property preferred_device:
        def __get__(Array self):
            return Device.wrapc(self.o.preferred_device())


    property strides:
        def __get__(Array self):
            return self.o.normalized_strides()

    def reshape(Array self, *args):
        cdef vector[int] new_shape
        if isinstance(args[0], int):
            new_shape = args
        else:
            new_shape = args[0]
        return Array.wrapc(self.o.reshape(new_shape))

    def __str__(Array self):
        cdef stringstream ss
        self.o.print_me(ss)
        return ss.to_string().decode("utf-8")

    def __repr__(Array self):
        return str(self)

