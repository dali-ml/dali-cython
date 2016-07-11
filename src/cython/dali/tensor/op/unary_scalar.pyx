cpdef Tensor scalar_add(Tensor a, scalar):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_add(a.o, <int>scalar))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_add(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_radd(scalar, Tensor a):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_add(<int>scalar, a.o))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_add(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_sub(Tensor a, scalar):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_sub(a.o, <int>scalar))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_sub(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_rsub(scalar, Tensor a):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_sub(<int>scalar, a.o))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_sub(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_mul(Tensor a, scalar):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_mul(a.o, <int>scalar))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_mul(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_rmul(scalar, Tensor a):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_mul(<int>scalar, a.o))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_mul(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))

cpdef Tensor scalar_div(Tensor a, scalar):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_div(a.o, <int>scalar))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_div(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_rdiv(scalar, Tensor a):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_div(<int>scalar, a.o))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_div(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_pow(Tensor a, scalar):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_pow(a.o, <int>scalar))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_pow(a.o, <float>scalar))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))


cpdef Tensor scalar_rpow(scalar, Tensor a):
    if type(scalar) == int:
        return Tensor.wrapc(c_scalar_pow(<int>scalar, a.o))
    elif type(scalar) == float:
        return Tensor.wrapc(c_scalar_pow(<float>scalar, a.o))
    else:
        raise ValueError("scalar must be int or float, got " + str(type(scalar)))
