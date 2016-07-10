cimport third_party.modern_numpy as c_np

from libcpp.string                     cimport string
from third_party.libcpp11.memory       cimport shared_ptr
from third_party.libcpp11.stringstream cimport stringstream
from third_party.libcpp11.vector       cimport vector

from .dtype                      cimport *
from .memory.device               import *
from .memory.device              cimport *
from .memory.synchronized_memory cimport *

cdef extern from "dali/array/array.h":
    cdef cppclass CArray "Array":
        CArray()
        CArray(vector[int], DType dtype, CDevice device) except +
        vector[int]& shape() except +
        vector[int] bshape()
        vector[int] subshape()
        bint is_transpose()
        bint spans_entire_memory() except +
        int offset() except +
        int number_of_elements()
        CArray swapaxes(int axis1, int axis2) except +
        CArray squeeze(int axis) except +
        int ndim() except +

        CArray reshape(const vector[int]&) except +
        void print_me "print" (stringstream& stream) const;
        DType dtype() except +
        CDevice preferred_device() except +
        shared_ptr[CSynchronizedMemory] memory() except +
        vector[int] normalized_strides() except +
        CArray transpose() except +
        CArray transpose(const vector[int]&) except +
        CArray dimshuffle(const vector[int]&) except +
        CArray ravel() except +
        CArray copyless_ravel() except +
        CArray copyless_reshape(const vector[int]&) except +

        CArray pluck_axis(const int& axis, const int& idx) except+
        CArray expand_dims(int new_axis) except+
        CArray broadcast_axis(int axis) except+
        CArray insert_broadcast_axis(int new_axis) except+
        CArray broadcast_scalar_to_ndim(const int&) except+

cdef class Array:
    """Multidimensional array of numbers.

    Parameters
    ----------
    shape: [int]
        a list representing sizes of subsequent dimensions
    dtype: np.dtype
        datatype used for representing numbers
    preferred_device: dali.Device
        preferred device for data storage. If it is equal to None,
        a dali.default_device() is used.
    """

    cdef CArray o

    @staticmethod
    cdef Array wrapc(CArray o)

    cdef c_np.NPY_TYPES cdtype(Array self)
