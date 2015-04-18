from libcpp.string cimport string
from libcpp.vector cimport vector

cdef string normalize_s(s):
    if type(s) is str:
        return s.encode("utf-8")
    elif type(s) is bytes:
        return s
    else:
        raise TypeError("Must pass a str or bytes object.")

include "dali/utils/core_utils.pyx"
include "dali/mat/Mat.pyx"
include "dali/mat/Layers.pyx"
