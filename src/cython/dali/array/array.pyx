from .array cimport CArray
from .dtype  cimport *

import numpy                     as np
cimport third_party.modern_numpy as np

cdef DType dtype_np_to_dali(dtype):
    if dtype == np.float32:
        return DTYPE_FLOAT
    elif dtype == np.float64:
        return DTYPE_DOUBLE
    elif dtype == np.int32:
        return DTYPE_INT32
    else:
        raise ValueError("Invalid dtype: " + str(dtype) +
                         " (Dali only supports np.int32, np.float32 or np.float64)")

cdef object dtype_dali_to_np(DType dtype):
    if dtype == DTYPE_FLOAT:
        return np.float32
    elif dtype == DTYPE_DOUBLE:
        return np.float64
    elif dtype == DTYPE_INT32:
        return np.int32
    else:
        raise Exception("Internal Dali Array dtype improperly set.")

cdef class Array:
    cdef CArray o

    def __cinit__(Array self, vector[int] shape, dtype=np.float32):
        self.o = CArray(shape, dtype_np_to_dali(dtype), CDevice.cpu())

    property dtype:
        def __get__(Array self):
            return dtype_dali_to_np(self.o.dtype())

    property shape:
        def __get__(Array self):
            return self.o.shape()

    property strides:
        def __get__(Array self):
            return self.o.normalized_strides()

    def reshape(Array self, *args):
        cdef vector[int] new_shape
        if isinstance(args[0], int):
            new_shape = args
        else:
            new_shape = args[0]
        ret = Array([])
        ret.o = self.o.reshape(new_shape)
        return ret

    def __str__(Array self):
        cdef stringstream ss
        self.o.print_me(ss)
        return ss.to_string().decode("utf-8")

    def __repr__(Array self):
        return str(self)
