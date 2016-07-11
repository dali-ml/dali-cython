from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/dropout.h" namespace "tensor_ops":
     CTensor c_dropout "tensor_ops::dropout" (const CTensor&, const double& drop_prob)
     CTensor c_dropout_unnormalized "tensor_ops::dropout_unnormalized" (const CTensor&, const double& drop_prob)
     CTensor c_fast_dropout "tensor_ops::fast_dropout" (const CTensor&)
