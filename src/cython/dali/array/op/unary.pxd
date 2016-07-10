from dali.array.array cimport *

cdef extern from "dali/array/op/unary.h" namespace "op" nogil:
    CAssignableArray c_sigmoid "op::sigmoid"(const CArray& a)
    CAssignableArray c_tanh "op::tanh"(const CArray& a)
    CAssignableArray c_relu "op::relu"(const CArray& a)
    CAssignableArray c_eltinv "op::eltinv"(const CArray& a)
    CAssignableArray c_exp "op::exp"(const CArray& a)
    CAssignableArray c_log "op::log"(const CArray& a)
    CAssignableArray c_log_or_zero "op::log_or_zero"(const CArray& a)
    CAssignableArray c_abs "op::abs"(const CArray& a)
    CAssignableArray c_sign "op::sign"(const CArray& a)
    CAssignableArray c_square "op::square"(const CArray& a)
    CAssignableArray c_softplus "op::softplus"(const CArray& a)
    CAssignableArray c_cube "op::cube"(const CArray& a)
    CAssignableArray c_sqrt "op::sqrt"(const CArray& a)
    CAssignableArray c_rsqrt "op::rsqrt"(const CArray& a)

cpdef AssignableArray sigmoid(Array a)
cpdef AssignableArray tanh(Array a)
cpdef AssignableArray relu(Array a)
cpdef AssignableArray eltinv(Array a)
cpdef AssignableArray exp(Array a)
cpdef AssignableArray log(Array a)
cpdef AssignableArray log_or_zero(Array a)
cpdef AssignableArray abs(Array a)
cpdef AssignableArray sign(Array a)
cpdef AssignableArray square(Array a)
cpdef AssignableArray softplus(Array a)
cpdef AssignableArray cube(Array a)
cpdef AssignableArray sqrt(Array a)
cpdef AssignableArray rsqrt(Array a)
