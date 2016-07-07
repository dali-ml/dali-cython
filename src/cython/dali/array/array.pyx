import numpy as np

from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free

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

    def get_value(self, copy=False):
        if copy:
            return np.array(self.get_value(False), copy=True)

        cdef np.ndarray ndarray
        cdef np.npy_intp* np_shape
        cdef vector[int] arr_shape
        arr_shape = self.o.shape()
        np_shape = <np.npy_intp*>malloc(self.o.ndim() * sizeof(np.npy_intp))
        try:
            for i in range(self.o.ndim()):
                np_shape[i] = <np.npy_intp>(arr_shape[i])
            ndarray = np.PyArray_SimpleNewFromData(
                self.o.ndim(),
                np_shape,
                np.dtype(self.dtype).num,
                self.o.memory().get()[0].mutable_data(CDevice.cpu())
            )
        finally:
            free(np_shape)

        ndarray.base = <PyObject*> self
        Py_INCREF(self)

        return ndarray

    def reshape(Array self, *args):
        cdef vector[int] new_shape
        if isinstance(args[0], int):
            new_shape = args
        else:
            new_shape = args[0]
        return Array.wrapc(self.o.reshape(new_shape))

    def debug_memory(Array self, bint print_contents=False):
        cdef stringstream ss
        cdef CSynchronizedMemory* mem = self.o.memory().get()
        mem[0].debug_info(ss, print_contents, self.o.dtype())
        return ss.to_string().decode("utf-8")

    def __str__(Array self):
        cdef stringstream ss
        self.o.print_me(ss)
        return ss.to_string().decode("utf-8")

    def __repr__(Array self):
        return str(self)


