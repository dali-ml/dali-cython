from dali.tensor.tensor cimport CTensor, Tensor
from dali.array.array cimport CArray, Array

cdef extern from "dali/tensor/op/solver_updates/adagrad.h" namespace "tensor_ops":
    void c_adagrad_update "tensor_ops::adagrad_update" (
        CTensor& t,
        CArray& cache,
        const double& step_size,
        const double& smooth_eps);
