from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/dot.h" namespace "tensor_ops":
    CTensor dot(const CTensor&, const CTensor&);
    CTensor vectordot(const CTensor& a, const CTensor& b);
    CTensor matrixdot(const CTensor& a, const CTensor& b);
    CTensor matrix_vector_dot(const CTensor& a, const CTensor& b);
    CTensor tensordot(const CTensor& a, const CTensor& b, const int& axis);
    CTensor tensordot(const CTensor& a, const CTensor& b,
                      const vector[int]& a_reduce_axes,
                      const vector[int]& b_reduce_axes);
