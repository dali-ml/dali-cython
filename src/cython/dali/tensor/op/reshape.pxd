from ..tensor                    cimport CTensor, Tensor, ensure_tensor_list
from third_party.libcpp11.vector cimport vector
from libcpp.string cimport string

cdef extern from "dali/tensor/op/reshape.h" namespace "tensor_ops":
    CTensor c_concatenate "tensor_ops::concatenate" (const vector[CTensor]& tensors, int axis) except +
    CTensor c_hstack "tensor_ops::hstack" (const vector[CTensor]& tensors) except +
    CTensor c_vstack "tensor_ops::vstack" (const vector[CTensor]& tensors) except +
    CTensor c_gather "tensor_ops::gather"(const CTensor& params, const CTensor& indices) except +

cpdef Tensor concatenate(tensors, int axis)
cpdef Tensor hstack(tensors)
cpdef Tensor vstack(tensors)
cpdef Tensor gather(Tensor params, Tensor indices)
