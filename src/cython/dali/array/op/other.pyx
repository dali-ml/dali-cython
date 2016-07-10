cpdef AssignableArray any_isinf(Array a, axis=None):
    if axis is None:
        return AssignableArray.wrapc(c_any_isinf(a.o))
    else:
        return AssignableArray.wrapc(c_any_isinf(a.o, axis))


cpdef AssignableArray any_isnan(Array a, axis=None):
    if axis is None:
        return AssignableArray.wrapc(c_any_isnan(a.o))
    else:
        return AssignableArray.wrapc(c_any_isnan(a.o, axis))

cpdef AssignableArray all_equals(Array left, Array right):
    return AssignableArray.wrapc(c_all_equals(left.o, right.o))

cpdef AssignableArray all_close(Array left, Array right, float tol):
    return AssignableArray.wrapc(c_all_close(left.o, right.o, tol))

cpdef AssignableArray argsort(Array a, axis=None):
    if axis is None:
        return AssignableArray.wrapc(c_argsort(a.o))
    else:
        return AssignableArray.wrapc(c_argsort(a.o, axis))

