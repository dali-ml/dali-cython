from libcpp.string                     cimport string
from third_party.libcpp11.vector       cimport vector
from third_party.libcpp11.stringstream cimport stringstream

from .dtype  cimport *
from .memory.device cimport *

cdef extern from "dali/array/array.h":
    cdef cppclass CArray "Array":
        CArray()
        CArray(vector[int], DType dtype, CDevice device) except +
        vector[int]& shape() except +

        CArray reshape(vector[int] new_shape) except +
        void print_me "print" (stringstream& stream) const;
        DType dtype() except +
        vector[int] normalized_strides() except +
