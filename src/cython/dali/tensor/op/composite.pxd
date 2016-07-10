from ..tensor                    cimport CTensor, Tensor, ensure_tensor_list
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/composite.h" namespace "tensor_ops::":
    CTensor cdot_with_bias "tensor_ops::dot_with_bias" (
        const CTensor& inputs,
        const CTensor& weight,
        const CTensor& bias)

    CTensor cmultiple_dot_with_bias "tensor_ops::multiple_dot_with_bias" (
        const vector[CTensor]& inputs,
        const vector[CTensor]& weights,
        CTensor bias)
    CTensor cquadratic_form "tensor_ops::quadratic_form"(
        const CTensor& left,
        const CTensor& middle,
        const CTensor& right)
