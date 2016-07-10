import numpy                     as np
cimport third_party.modern_numpy as c_np

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

cdef inline DType dtype_c_np_to_dali(c_np.NPY_TYPES dtype):
    if dtype == c_np.NPY_FLOAT32:
        return DTYPE_FLOAT
    elif dtype == c_np.NPY_FLOAT64:
        return DTYPE_DOUBLE
    elif dtype == c_np.NPY_INT32:
        return DTYPE_INT32
    else:
        raise ValueError("Invalid c_np dtype: " + str(dtype) +
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

cdef inline c_np.NPY_TYPES dtype_dali_to_c_np(DType dtype):
    if dtype == DTYPE_FLOAT:
        return c_np.NPY_FLOAT32
    elif dtype == DTYPE_DOUBLE:
        return c_np.NPY_FLOAT64
    elif dtype == DTYPE_INT32:
        return c_np.NPY_INT32
    else:
        raise Exception("Internal Dali Array dtype improperly set.")

cdef inline int dtype_to_itemsize(DType dtype):
    if dtype == DTYPE_FLOAT:
        return sizeof(float)
    elif dtype == DTYPE_DOUBLE:
        return sizeof(double)
    elif dtype == DTYPE_INT32:
        return sizeof(int)
    else:
        raise Exception("Internal Dali Array dtype improperly set.")

cdef inline bint is_cnp_dtype_supported(c_np.NPY_TYPES dtype):
    return dtype == c_np.NPY_FLOAT32 or dtype == c_np.NPY_FLOAT64 or dtype == c_np.NPY_INT32
