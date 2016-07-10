from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/binary.h" namespace "tensor_ops":
    CTensor cadd "tensor_ops::add"(const CTensor&, const CTensor&)
    CTensor cadd "tensor_ops::add"(const vector[CTensor]& tensors)
    CTensor csub "tensor_ops::sub"(const CTensor&, const CTensor&)
    CTensor celtmul "tensor_ops::eltmul"(const CTensor&, const CTensor&)
    CTensor celtdiv "tensor_ops::eltdiv"(const CTensor&, const CTensor&)
    CTensor cpow "tensor_ops::pow"(const CTensor&, const CTensor& exponent)
    CTensor ccircular_convolution "tensor_ops::circular_convolution"(const CTensor& content, const CTensor& shift)
    CTensor cprelu "tensor_ops::prelu"(const CTensor& x, const CTensor& weights)
