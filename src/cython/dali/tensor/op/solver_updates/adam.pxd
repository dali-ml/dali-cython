from dali.tensor.tensor cimport CTensor, Tensor
from dali.array.array cimport CArray, Array

cdef extern from "dali/tensor/op/solver_updates/adam.h" namespace "tensor_ops":
    void c_adam_update "tensor_ops::adam_update" (
        CTensor& param,
        CArray& m,
        CArray& v,
        const double& b1,
        const double& b2,
        const double& smooth_eps,
        const double& step_size,
        unsigned long long epoch);
