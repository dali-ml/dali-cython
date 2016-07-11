from dali.tensor.tensor cimport CTensor, Tensor
from dali.array.array cimport CArray, Array

cdef extern from "dali/tensor/op/solver_updates/adadelta.h" namespace "tensor_ops":
    void c_adadelta_update "tensor_ops::adadelta_update" (
        CTensor& param,
        CArray& gsum,
        CArray& xsum,
        const double& rho,
        const double& smooth_eps);
