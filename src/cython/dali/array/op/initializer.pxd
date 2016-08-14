from dali.array.array            cimport *
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/array/op/initializer.h" namespace "initializer":
    CAssignableArray initializer_empty "initializer::empty"()
    CAssignableArray initializer_zeros "initializer::zeros"()
    CAssignableArray initializer_ones "initializer::ones"()
    CAssignableArray initializer_arange "initializer::arange"(const double& start, const double& step) except +
    CAssignableArray initializer_fill_double "initializer::fill" (const double& constant)
    CAssignableArray initializer_fill_int "initializer::fill" (const int& constant)
    CAssignableArray initializer_gaussian "initializer::gaussian"(const double& mean, const double& std)
    CAssignableArray initializer_uniform "initializer::uniform"(const double& low, const double& high) except +
    CAssignableArray initializer_bernoulli "initializer::bernoulli"(const double& prob) except +
    CAssignableArray initializer_bernoulli_normalized "initializer::bernoulli_normalized"(const double& prob) except +
    CAssignableArray initializer_eye "initializer::eye"(const double& diag)
