cpdef AssignableArray scalar_add(Array a, scalar):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_add(a.o, <int>scalar))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_add(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_radd(scalar, Array a):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_add(<int>scalar, a.o))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_add(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_sub(Array a, scalar):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_sub(a.o, <int>scalar))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_sub(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_rsub(scalar, Array a):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_sub(<int>scalar, a.o))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_sub(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_mul(Array a, scalar):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_mul(a.o, <int>scalar))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_mul(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_rmul(scalar, Array a):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_mul(<int>scalar, a.o))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_mul(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))

cpdef AssignableArray scalar_div(Array a, scalar):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_div(a.o, <int>scalar))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_div(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_rdiv(scalar, Array a):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_div(<int>scalar, a.o))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_div(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_pow(Array a, scalar):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_pow(a.o, <int>scalar))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_pow(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef AssignableArray scalar_rpow(scalar, Array a):
    if type(scalar) == int:
        return AssignableArray.wrapc(c_scalar_pow(<int>scalar, a.o))
    elif type(scalar) == float:
        return AssignableArray.wrapc(c_scalar_pow(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))
