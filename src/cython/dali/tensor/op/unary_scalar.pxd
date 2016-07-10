from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/unary.h" namespace "tensor_ops":
    CTensor scalar_add(const CTensor&, const double& scalar)
    CTensor scalar_sub(const CTensor&, const double& scalar)
    CTensor scalar_eltmul(const CTensor&, const double& scalar)
    CTensor scalar_div(const CTensor&, const double& scalar)
    CTensor scalar_pow(const CTensor&, const double& scalar)

    CTensor scalar_add(const CTensor&, const float& scalar)
    CTensor scalar_sub(const CTensor&, const float& scalar)
    CTensor scalar_eltmul(const CTensor&, const float& scalar)
    CTensor scalar_div(const CTensor&, const float& scalar)
    CTensor scalar_pow(const CTensor&, const float& scalar)

    CTensor scalar_add(const CTensor&, const int& scalar)
    CTensor scalar_sub(const CTensor&, const int& scalar)
    CTensor scalar_eltmul(const CTensor&, const int& scalar)
    CTensor scalar_div(const CTensor&, const int& scalar)
    CTensor scalar_pow(const CTensor&, const int& scalar)

    CTensor scalar_add(const double&, const CTensor& scalar)
    CTensor scalar_sub(const double&, const CTensor& scalar)
    CTensor scalar_eltmul(const double&, const CTensor& scalar)
    CTensor scalar_div(const double&, const CTensor& scalar)
    CTensor scalar_pow(const double&, const CTensor& scalar)

    CTensor scalar_add(const float&, const CTensor& scalar)
    CTensor scalar_sub(const float&, const CTensor& scalar)
    CTensor scalar_eltmul(const float&, const CTensor& scalar)
    CTensor scalar_div(const float&, const CTensor& scalar)
    CTensor scalar_pow(const float&, const CTensor& scalar)

    CTensor scalar_add(const int&, const CTensor& scalar)
    CTensor scalar_sub(const int&, const CTensor& scalar)
    CTensor scalar_eltmul(const int&, const CTensor& scalar)
    CTensor scalar_div(const int&, const CTensor& scalar)
    CTensor scalar_pow(const int&, const CTensor& scalar)
