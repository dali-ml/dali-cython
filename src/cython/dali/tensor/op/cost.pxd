from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/cost.h" namespace "tensor_ops":
    CTensor binary_cross_entropy(const CTensor&, const double& target)
    CTensor binary_cross_entropy(const CTensor&, const CTensor&)
    CTensor sigmoid_binary_cross_entropy(const CTensor&, const double& target)
    CTensor sigmoid_binary_cross_entropy(const CTensor&, const CTensor&)
    CTensor margin_loss(const CTensor&, const int& target, const double& margin, const int& axis)
    CTensor margin_loss(const CTensor&, const CTensor& target, const double& margin, const int& axis)
    CTensor softmax_cross_entropy(const CTensor& unnormalized_probs,
                                  const CTensor& targets,
                                  const double& temperature,
                                  int axis)
    CTensor softmax(const CTensor&, int axis, const double& temperature)
    CTensor cross_entropy(const CTensor& probs, const CTensor& target, int axis)
