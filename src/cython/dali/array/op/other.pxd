from dali.array.array cimport *

cdef extern from "dali/array/op/other.h" namespace "op" nogil:
    CAssignableArray c_any_isnan "op::any_isnan"(const CArray& a)
    CAssignableArray c_any_isinf "op::any_isinf"(const CArray& a)
    CAssignableArray c_any_isnan "op::any_isnan"(const CArray& a, int axis)
    CAssignableArray c_any_isinf "op::any_isinf"(const CArray& a, int axis)
    CAssignableArray c_all_equals "op::all_equals"(const CArray& left, const CArray& right)
    CAssignableArray c_all_close "op::all_close"(const CArray& left, const CArray& right, const double& atolerance)
    CAssignableArray c_argsort "op::argsort"(const CArray& a, int axis)
    CAssignableArray c_argsort "op::argsort"(const CArray& a)

cpdef AssignableArray any_isinf(Array a, axis=?)
cpdef AssignableArray any_isnan(Array a, axis=?)
cpdef AssignableArray all_equals(Array left, Array right)
cpdef AssignableArray all_close(Array left, Array right, float tol)
cpdef AssignableArray argsort(Array a, axis=?)
