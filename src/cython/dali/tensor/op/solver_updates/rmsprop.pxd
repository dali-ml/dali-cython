from dali.tensor.tensor cimport CTensor, Tensor
from dali.array.array cimport CArray, Array

cdef extern from "dali/tensor/op/solver_updates/adam.h" namespace "tensor_ops":
    void c_rmsprop_update "tensor_ops::rmsprop_update" (
        CTensor& param,
        CArray& cache,
        const double& decay_rate,
        const double& step_size,
        const double& smooth_eps);

    void c_rmsprop_momentum_update "tensor_ops::rmsprop_momentum_update" (
        CTensor& param,
        CArray& n_cache,
        CArray& g_cache,
        CArray& momentum_cache,
        const double& decay_rate,
        const double& momentum,
        const double& step_size,
        const double& smooth_eps);
