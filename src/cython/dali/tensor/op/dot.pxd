from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/dot.h" namespace "tensor_ops":
    CTensor cdot "tensor_ops::dot" (
        const CTensor&, const CTensor&) except +;
    CTensor couter "tensor_ops::outer" (
        const CTensor&, const CTensor&) except +;
    CTensor cinner "tensor_ops::inner" (
        const CTensor& a, const CTensor& b) except +;
    CTensor cmatrixdot "tensor_ops::matrixdot" (
        const CTensor& a, const CTensor& b) except +;
    CTensor cmatrix_vector_dot "tensor_ops::matrix_vector_dot" (
        const CTensor& a, const CTensor& b) except +;
    CTensor ctensordot "tensor_ops::tensordot" (
        const CTensor& a,
        const CTensor& b,
        const int& axis);
    CTensor ctensordot "tensor_ops::tensordot" (
        const CTensor& a,
        const CTensor& b,
        const vector[int]& a_reduce_axes,
        const vector[int]& b_reduce_axes) except +;
