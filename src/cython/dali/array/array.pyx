import numpy as np

from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
c_np.import_array()

cdef object list_from_args(args):
    if len(args) > 0:
        if all(isinstance(arg, int) for arg in args):
            return args
        if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
            return args[0]
    raise ValueError("expected a list of integers")

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

    cdef c_np.NPY_TYPES cdtype(Array self):
        return dtype_dali_to_c_np(self.o.dtype())

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

    def transpose(Array self, *dims):
        cdef vector[int] cdims = list_from_args(dims)
        return Array.wrapc(self.o.transpose(cdims))

    def get_value(self, copy=False):
        if copy:
            return np.array(self.get_value(False), copy=True)
        cdef c_np.ndarray ndarray
        cdef c_np.npy_intp* np_shape = <c_np.npy_intp*>malloc(self.o.ndim() * sizeof(c_np.npy_intp))
        cdef vector[int] arr_shape = self.o.shape()
        cdef vector[int] strides = self.o.normalized_strides()

        try:
            for i in range(self.o.ndim()):
                np_shape[i] = <c_np.npy_intp>(arr_shape[i])
            ndarray = c_np.PyArray_SimpleNewFromData(
                self.o.ndim(),
                np_shape,
                self.cdtype(),
                self.o.memory().get()[0].mutable_data(CDevice.cpu())
            )

            for i in range(self.o.ndim()):
                ndarray.strides[i] = strides[i] * dtype_to_itemsize(self.o.dtype())
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
        return "Array(" + ss.to_string().decode("utf-8") + ")"

    def __repr__(Array self):
        return str(self)


