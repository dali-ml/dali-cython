from ..tensor                    cimport CTensor, Tensor, ensure_tensor_list
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

cpdef Tensor add_n(tensors)
cpdef Tensor add(Tensor left, Tensor right)
cpdef Tensor sub(Tensor left, Tensor right)
cpdef Tensor eltmul(Tensor left, Tensor right)
cpdef Tensor eltdiv(Tensor left, Tensor right)
cpdef Tensor pow(Tensor x, Tensor exponent)
cpdef Tensor circular_convolution(Tensor content, Tensor shift)
cpdef Tensor prelu(Tensor x, Tensor weights)
