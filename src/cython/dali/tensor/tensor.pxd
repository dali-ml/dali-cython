from ..array.array               cimport CArray, Array
from ..array.dtype               cimport *
from ..array.memory.device        import *
from ..array.memory.device       cimport *

from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/tensor.h":
    cdef cppclass CTensor "Tensor":
        bint constant
        CArray w
        CArray dw

        CTensor()
        CTensor(const vector[int]& shape, DType dtype, CDevice device)
        CTensor(const CArray& other, bint copy)

        DType dtype() except +
        CDevice preferred_device() except +

        void grad() except +
        void clear_grad() except +
        void clear() except +
        vector[int]& shape()

        bint is_stateless() except +
        bint is_scalar() except +
        bint is_vector() except +
        bint is_matrix() except +
        bint is_nan() except +
        bint is_grad_nan() except +

        int ndim()
        int number_of_elements()

        CTensor sum(const int& axis) except +
        CTensor sum() except +
        CTensor mean(const int& axis) except +
        CTensor mean() except +
        CTensor max(const int& axis) except +
        CTensor max() except +
        CTensor min(const int& axis) except +
        CTensor min() except +
        CTensor L2_norm(const int& axis) except +
        CTensor L2_norm() except +
        CTensor log() except +
        CTensor exp() except +
        CTensor abs() except +
        CTensor tanh() except +
        CTensor softplus() except +
        CTensor relu() except +
        CTensor dot(const CTensor&) except +
        CTensor sqrt() except +
        CTensor square() except +
        CTensor eltinv() except +
        CTensor sigmoid() except +
        CTensor steep_sigmoid() except +
        CTensor steep_sigmoid(const double&) except +

        CTensor swapaxes(int axis1, int axis2) except +

        CTensor reshape(const vector[int]&) except +
        CTensor copyless_reshape(const vector[int]&) except +

        CTensor ravel() except +
        CTensor copyless_ravel() except +

        @staticmethod
        CTensor zeros(const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor ones(const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor empty(const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor arange(const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor arange_double "arange"(const double& start, const double& stop, const double& step, DType dtype, CDevice device) except +
        @staticmethod
        CTensor uniform(double limit, const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor uniform(double low, double high, const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor gaussian(double mean, double std, const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor bernoulli(double prob, const vector[int]& shape, DType dtype, CDevice device) except +
        @staticmethod
        CTensor bernoulli_normalized(double prob, const vector[int]& shape, DType dtype, CDevice device) except +

cpdef Tensor ensure_tensor(object arr)
cdef vector[CTensor] ensure_tensor_list(object tensors)

cdef class Tensor:
    """
    Multidimensional array of numbers that keeps track
    of gradients to perform automatic differentiation.

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

    cdef CTensor o

    @staticmethod
    cdef Tensor wrapc(CTensor o)

    cdef c_np.NPY_TYPES cdtype(Tensor self)
