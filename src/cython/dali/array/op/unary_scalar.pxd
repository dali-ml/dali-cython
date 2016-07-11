from dali.array.array            cimport *
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/array/op/unary_scalar.h" namespace "op":
    CAssignableArray c_scalar_add "op::scalar_add"(const CArray&, const double& scalar)
    CAssignableArray c_scalar_sub "op::scalar_sub"(const CArray&, const double& scalar)
    CAssignableArray c_scalar_mul "op::scalar_mul"(const CArray&, const double& scalar)
    CAssignableArray c_scalar_div "op::scalar_div"(const CArray&, const double& scalar)
    CAssignableArray c_scalar_pow "op::scalar_pow"(const CArray&, const double& scalar)

    # in theory in python this is never needed
    CAssignableArray c_scalar_add "op::scalar_add"(const CArray&, const float& scalar)
    CAssignableArray c_scalar_sub "op::scalar_sub"(const CArray&, const float& scalar)
    CAssignableArray c_scalar_mul "op::scalar_mul"(const CArray&, const float& scalar)
    CAssignableArray c_scalar_div "op::scalar_div"(const CArray&, const float& scalar)
    CAssignableArray c_scalar_pow "op::scalar_pow"(const CArray&, const float& scalar)

    CAssignableArray c_scalar_add "op::scalar_add"(const CArray&, const int& scalar)
    CAssignableArray c_scalar_sub "op::scalar_sub"(const CArray&, const int& scalar)
    CAssignableArray c_scalar_mul "op::scalar_mul"(const CArray&, const int& scalar)
    CAssignableArray c_scalar_div "op::scalar_div"(const CArray&, const int& scalar)
    CAssignableArray c_scalar_pow "op::scalar_pow"(const CArray&, const int& scalar)

    CAssignableArray c_scalar_add "op::scalar_add"(const double&, const CArray& scalar)
    CAssignableArray c_scalar_sub "op::scalar_sub"(const double&, const CArray& scalar)
    CAssignableArray c_scalar_mul "op::scalar_mul"(const double&, const CArray& scalar)
    CAssignableArray c_scalar_div "op::scalar_div"(const double&, const CArray& scalar)
    CAssignableArray c_scalar_pow "op::scalar_pow"(const double&, const CArray& scalar)

    CAssignableArray c_scalar_add "op::scalar_add"(const float&, const CArray& scalar)
    CAssignableArray c_scalar_sub "op::scalar_sub"(const float&, const CArray& scalar)
    CAssignableArray c_scalar_mul "op::scalar_mul"(const float&, const CArray& scalar)
    CAssignableArray c_scalar_div "op::scalar_div"(const float&, const CArray& scalar)
    CAssignableArray c_scalar_pow "op::scalar_pow"(const float&, const CArray& scalar)

    CAssignableArray c_scalar_add "op::scalar_add"(const int&, const CArray& scalar)
    CAssignableArray c_scalar_sub "op::scalar_sub"(const int&, const CArray& scalar)
    CAssignableArray c_scalar_mul "op::scalar_mul"(const int&, const CArray& scalar)
    CAssignableArray c_scalar_div "op::scalar_div"(const int&, const CArray& scalar)
    CAssignableArray c_scalar_pow "op::scalar_pow"(const int&, const CArray& scalar)


cpdef AssignableArray scalar_add(Array a, scalar)
cpdef AssignableArray scalar_radd(scalar, Array a)

cpdef AssignableArray scalar_sub(Array a, scalar)
cpdef AssignableArray scalar_rsub(scalar, Array a)

cpdef AssignableArray scalar_mul(Array a, scalar)
cpdef AssignableArray scalar_rmul(scalar, Array a)

cpdef AssignableArray scalar_div(Array a, scalar)
cpdef AssignableArray scalar_rdiv(scalar, Array a)

cpdef AssignableArray scalar_pow(Array a, scalar)
cpdef AssignableArray scalar_rpow(scalar, Array a)
