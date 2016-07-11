from dali.tensor.tensor cimport CTensor, Tensor
from dali.array.array cimport CArray, Array

cdef extern from "dali/tensor/op/solver_updates/clip_and_regularize.h" namespace "tensor_ops":
    void c_clip_and_regularize "tensor_ops::clip_and_regularize" (
        const CTensor& param,
        const double& clip_abs,
        const double& clip_norm,
        const double& regc);

    void c_regularize "tensor_ops::regularize" (
        const CTensor& param,
        const double& regc);

    void c_normalize_gradient "tensor_ops::normalize_gradient" (
        const CTensor& param,
        const double& norm_threshold);
