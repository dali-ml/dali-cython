cimport third_party.modern_numpy as c_np

from libcpp.string                     cimport string
from third_party.libcpp11.memory       cimport shared_ptr
from third_party.libcpp11.stringstream cimport stringstream
from third_party.libcpp11.vector       cimport vector

from .dtype                      cimport *
from .memory.device               import *
from .memory.device              cimport *
from .memory.synchronized_memory cimport *
from .slice                      cimport *

cdef extern from "dali/array/array.h":
    cdef cppclass CArray "Array":
        CArray()
        CArray(vector[int], DType dtype, CDevice device) except +
        vector[int]& shape() except +
        vector[int] bshape()
        vector[int] subshape()
        bint is_transpose()
        bint spans_entire_memory() except +
        bint contiguous_memory() except +
        int offset() except +
        int number_of_elements()
        CArray swapaxes(int axis1, int axis2) except +
        CArray squeeze(int axis) except +
        int ndim() except +
        void clear() except+

        @staticmethod
        CArray arange(const vector[int]& shape, DType dtype, CDevice preferred_device);
        @staticmethod
        CArray zeros(const vector[int]& shape, DType dtype, CDevice preferred_device);
        @staticmethod
        CArray zeros_like(const CArray& other);
        @staticmethod
        CArray empty_like(const CArray& other);
        @staticmethod
        CArray ones(const vector[int]& shape, DType dtype, CDevice preferred_device);
        @staticmethod
        CArray ones_like(const CArray& other);


        void to_device(CDevice device) except +

        CArray reshape(const vector[int]&) except +
        void print_me "print" (stringstream& stream) except +
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
        @staticmethod
        CArray adopt_buffer(void* buffer,
                            const vector[int]& shape,
                            DType dtype,
                            CDevice buffer_location,
                            const vector[int]& strides) except +
        void copy_from(const CArray& other) except +

        CArray pluck_axis(const int& axis, const int& idx) except+
        CArray expand_dims(int new_axis) except+
        CArray broadcast_axis(int axis) except+
        CArray insert_broadcast_axis(int new_axis) except+
        CArray broadcast_scalar_to_ndim(const int&) except+

        CArray operator_bracket "operator[]"(const int&) except +
        CSlicingInProgressArray operator_bracket "operator[]"(const CSlice&)  except +
        CSlicingInProgressArray operator_bracket "operator[]"(const CBroadcast&)  except +

    cdef cppclass CAssignableArray "Assignable<Array>":
        CAssignableArray "Assignable<Array>"()

        CArray eval()


cdef class AssignableArray:
    cdef CAssignableArray o

    @staticmethod
    cdef AssignableArray wrapc(CAssignableArray o) except +

    cpdef Array eval(AssignableArray self)

cpdef Array ensure_array(object arr)

cdef class Array:
    """Array(data, dtype=None, preferred_device=None, borrow=False)

    Multidimensional array of numbers.

    Parameters
    ----------
    data: list, tuple or np.array
        numbers to be crunched.
    dtype: np.dtype
        if dtype is not None, the array is converted to this dtype.
    preferred_device: dali.Device
        preferred device for data storage. If it is equal to None,
        a dali.default_device() is used.
    borrow: bool
        if true, Array will attempt to create a view onto the
        data and steal the ownership as a sideeffect.
    """

    cdef CArray o

    @staticmethod
    cdef Array wrapc(CArray o) except +

    cdef void use_numpy_memory(Array self,
                               c_np.ndarray py_array,
                               c_np.NPY_TYPES dtype,
                               CDevice preferred_device,
                               bint steal) except +

    cdef c_np.NPY_TYPES cdtype(Array self) except +


