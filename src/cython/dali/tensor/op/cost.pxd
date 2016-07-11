from ..tensor                    cimport CTensor, Tensor, ensure_tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/cost.h" namespace "tensor_ops":
    CTensor cbinary_cross_entropy_double "tensor_ops::binary_cross_entropy" (
        const CTensor&,
        const double& target) except +
    CTensor cbinary_cross_entropy "tensor_ops::binary_cross_entropy" (
        const CTensor&,
        const CTensor&) except +
    CTensor csigmoid_binary_cross_entropy_double "tensor_ops::sigmoid_binary_cross_entropy" (
        const CTensor&,
        const double& target) except +
    CTensor csigmoid_binary_cross_entropy "tensor_ops::sigmoid_binary_cross_entropy" (
        const CTensor&,
        const CTensor&) except +
    CTensor cmargin_loss_int "tensor_ops::margin_loss" (
        const CTensor&,
        const int& target,
        const double& margin,
        const int& axis) except +
    CTensor cmargin_loss "tensor_ops::margin_loss" (
        const CTensor&,
        const CTensor& target,
        const double& margin,
        const int& axis) except +
    CTensor csoftmax_cross_entropy "tensor_ops::softmax_cross_entropy" (
        const CTensor& unnormalized_probs,
        const CTensor& targets,
        const double& temperature,
        int axis) except +
    CTensor csoftmax "tensor_ops::softmax" (
        const CTensor&,
        int axis,
        const double& temperature) except +
    CTensor ccross_entropy "tensor_ops::cross_entropy" (
        const CTensor& probs,
        const CTensor& target,
        int axis) except +

cpdef Tensor binary_cross_entropy(Tensor t, target)
cpdef Tensor binary_cross_entropy(Tensor t, target)
cpdef Tensor margin_loss(Tensor t, target, double margin, int axis)
cpdef Tensor softmax_cross_entropy(Tensor logits, Tensor target, double temperature=?, int axis=?)
cpdef Tensor softmax(Tensor t, int axis=?, double temperature=?)
cpdef Tensor cross_entropy(Tensor probs, Tensor target, int axis=?)
