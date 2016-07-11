from dali.tensor.tensor cimport CTensor, Tensor
from dali.array.array cimport CArray, Array

cdef extern from "dali/tensor/op/solver_updates/sgd.h" namespace "tensor_ops":
    void c_sgd_update "tensor_ops::sgd_update" (
    	CTensor& t, const double& step_size);
