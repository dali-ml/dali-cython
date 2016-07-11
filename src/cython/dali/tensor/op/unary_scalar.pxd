from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/unary_scalar.h" namespace "tensor_ops":
    CTensor c_scalar_add "tensor_ops::scalar_add"(const CTensor&, const double& scalar)
    CTensor c_scalar_sub "tensor_ops::scalar_sub"(const CTensor&, const double& scalar)
    CTensor c_scalar_mul "tensor_ops::scalar_mul"(const CTensor&, const double& scalar)
    CTensor c_scalar_div "tensor_ops::scalar_div"(const CTensor&, const double& scalar)
    CTensor c_scalar_pow "tensor_ops::scalar_pow"(const CTensor&, const double& scalar)

    # in theory in python this is never needed
    CTensor c_scalar_add "tensor_ops::scalar_add"(const CTensor&, const float& scalar)
    CTensor c_scalar_sub "tensor_ops::scalar_sub"(const CTensor&, const float& scalar)
    CTensor c_scalar_mul "tensor_ops::scalar_mul"(const CTensor&, const float& scalar)
    CTensor c_scalar_div "tensor_ops::scalar_div"(const CTensor&, const float& scalar)
    CTensor c_scalar_pow "tensor_ops::scalar_pow"(const CTensor&, const float& scalar)

    CTensor c_scalar_add "tensor_ops::scalar_add"(const CTensor&, const int& scalar)
    CTensor c_scalar_sub "tensor_ops::scalar_sub"(const CTensor&, const int& scalar)
    CTensor c_scalar_mul "tensor_ops::scalar_mul"(const CTensor&, const int& scalar)
    CTensor c_scalar_div "tensor_ops::scalar_div"(const CTensor&, const int& scalar)
    CTensor c_scalar_pow "tensor_ops::scalar_pow"(const CTensor&, const int& scalar)

    CTensor c_scalar_add "tensor_ops::scalar_add"(const double&, const CTensor& scalar)
    CTensor c_scalar_sub "tensor_ops::scalar_sub"(const double&, const CTensor& scalar)
    CTensor c_scalar_mul "tensor_ops::scalar_mul"(const double&, const CTensor& scalar)
    CTensor c_scalar_div "tensor_ops::scalar_div"(const double&, const CTensor& scalar)
    CTensor c_scalar_pow "tensor_ops::scalar_pow"(const double&, const CTensor& scalar)

    CTensor c_scalar_add "tensor_ops::scalar_add"(const float&, const CTensor& scalar)
    CTensor c_scalar_sub "tensor_ops::scalar_sub"(const float&, const CTensor& scalar)
    CTensor c_scalar_mul "tensor_ops::scalar_mul"(const float&, const CTensor& scalar)
    CTensor c_scalar_div "tensor_ops::scalar_div"(const float&, const CTensor& scalar)
    CTensor c_scalar_pow "tensor_ops::scalar_pow"(const float&, const CTensor& scalar)

    CTensor c_scalar_add "tensor_ops::scalar_add"(const int&, const CTensor& scalar)
    CTensor c_scalar_sub "tensor_ops::scalar_sub"(const int&, const CTensor& scalar)
    CTensor c_scalar_mul "tensor_ops::scalar_mul"(const int&, const CTensor& scalar)
    CTensor c_scalar_div "tensor_ops::scalar_div"(const int&, const CTensor& scalar)
    CTensor c_scalar_pow "tensor_ops::scalar_pow"(const int&, const CTensor& scalar)


cpdef Tensor scalar_add(Tensor a, scalar)
cpdef Tensor scalar_radd(scalar, Tensor a)

cpdef Tensor scalar_sub(Tensor a, scalar)
cpdef Tensor scalar_rsub(scalar, Tensor a)

cpdef Tensor scalar_mul(Tensor a, scalar)
cpdef Tensor scalar_rmul(scalar, Tensor a)

cpdef Tensor scalar_div(Tensor a, scalar)
cpdef Tensor scalar_rdiv(scalar, Tensor a)

cpdef Tensor scalar_pow(Tensor a, scalar)
cpdef Tensor scalar_rpow(scalar, Tensor a)
