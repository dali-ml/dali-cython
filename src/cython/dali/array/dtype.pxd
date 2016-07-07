import numpy                     as np
cimport third_party.modern_numpy as np

cdef extern from "dali/array/dtype.h":
    enum DType:
        DTYPE_FLOAT  = 0
        DTYPE_DOUBLE = 1
        DTYPE_INT32  = 2

cdef inline DType dtype_np_to_dali(dtype):
    if dtype == np.float32:
        return DTYPE_FLOAT
    elif dtype == np.float64:
        return DTYPE_DOUBLE
    elif dtype == np.int32:
        return DTYPE_INT32
    else:
        raise ValueError("Invalid dtype: " + str(dtype) +
                         " (Dali only supports np.int32, np.float32 or np.float64)")

cdef inline object dtype_dali_to_np(DType dtype):
    if dtype == DTYPE_FLOAT:
        return np.float32
    elif dtype == DTYPE_DOUBLE:
        return np.float64
    elif dtype == DTYPE_INT32:
        return np.int32
    else:
        raise Exception("Internal Dali Array dtype improperly set.")
