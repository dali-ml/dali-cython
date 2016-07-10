from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/composite.h" namespace "tensor_ops":
	CTensor dot_with_bias(const CTensor& inputs,
                          const CTensor& weight,
                          const CTensor& bias)

    CTensor multiple_dot_with_bias(const vector[CTensor]& inputs,
                                   const vector[CTensor]& weights,
                                   CTensor bias)
    CTensor quadratic_form(const CTensor& left, const CTensor& middle, const CTensor& right)
