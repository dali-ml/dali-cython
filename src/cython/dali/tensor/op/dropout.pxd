from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/dropout.h" namespace "tensor_ops":
     CTensor dropout(const CTensor&, const double& drop_prob)
     CTensor dropout_unnormalized(const CTensor&, const double& drop_prob)
     CTensor fast_dropout(const CTensor&)
