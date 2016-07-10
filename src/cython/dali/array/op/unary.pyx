from ..array cimport AssignableArray, Array, CAssignableArray, CArray

cpdef AssignableArray sigmoid(Array a):
    return AssignableArray.wrapc(c_sigmoid(a.o))

cpdef AssignableArray tanh(Array a):
    return AssignableArray.wrapc(c_tanh(a.o))

cpdef AssignableArray relu(Array a):
    return AssignableArray.wrapc(c_relu(a.o))

cpdef AssignableArray eltinv(Array a):
    return AssignableArray.wrapc(c_eltinv(a.o))

cpdef AssignableArray exp(Array a):
    return AssignableArray.wrapc(c_exp(a.o))

cpdef AssignableArray log(Array a):
    return AssignableArray.wrapc(c_log(a.o))

cpdef AssignableArray log_or_zero(Array a):
    return AssignableArray.wrapc(c_log_or_zero(a.o))

cpdef AssignableArray abs(Array a):
    return AssignableArray.wrapc(c_abs(a.o))

cpdef AssignableArray sign(Array a):
    return AssignableArray.wrapc(c_sign(a.o))

cpdef AssignableArray square(Array a):
    return AssignableArray.wrapc(c_square(a.o))

cpdef AssignableArray softplus(Array a):
    return AssignableArray.wrapc(c_softplus(a.o))

cpdef AssignableArray cube(Array a):
    return AssignableArray.wrapc(c_cube(a.o))

cpdef AssignableArray sqrt(Array a):
    return AssignableArray.wrapc(c_sqrt(a.o))

cpdef AssignableArray rsqrt(Array a):
    return AssignableArray.wrapc(c_rsqrt(a.o))
