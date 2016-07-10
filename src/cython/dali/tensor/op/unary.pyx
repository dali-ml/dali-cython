
cpdef Tensor tanh(Tensor t):
    return Tensor.wrapc(ctanh(t.o))

cpdef Tensor softplus(Tensor t):
    return Tensor.wrapc(csoftplus(t.o))

cpdef Tensor abs(Tensor t):
    return Tensor.wrapc(cabs(t.o))

cpdef Tensor log(Tensor t):
    return Tensor.wrapc(clog(t.o))

# clipped relu: relu(x ; clip) = max( min(clip, x), 0 );
cpdef Tensor relu(Tensor t, upper_bound=None):
    if upper_bound is None:
        return Tensor.wrapc(crelu(t.o))
    else:
        return Tensor.wrapc(crelu(t.o, upper_bound))

cpdef Tensor exp(Tensor t):
    return Tensor.wrapc(cexp(t.o))

cpdef Tensor sigmoid(Tensor t):
    return Tensor.wrapc(csigmoid(t.o))

cpdef Tensor eltinv(Tensor t):
    return Tensor.wrapc(celtinv(t.o))

cpdef Tensor sqrt(Tensor t):
    return Tensor.wrapc(csqrt(t.o))

cpdef Tensor square(Tensor t):
    return Tensor.wrapc(csquare(t.o))

cpdef Tensor cube(Tensor t):
    return Tensor.wrapc(ccube(t.o))

cpdef Tensor rsqrt(Tensor t):
    return Tensor.wrapc(crsqrt(t.o))

cpdef Tensor eltmax(Tensor t, double lower_bound):
    return Tensor.wrapc(celtmax(t.o, lower_bound))

cpdef Tensor eltmin(Tensor t, double upper_bound):
    return Tensor.wrapc(celtmin(t.o, upper_bound))

cpdef Tensor steep_sigmoid(Tensor t, double agressiveness=3.75):
    return Tensor.wrapc(csteep_sigmoid(t.o, agressiveness))

cpdef Tensor relu100(Tensor t):
    return Tensor.wrapc(crelu100(t.o))

cpdef Tensor relu20(Tensor t):
    return Tensor.wrapc(crelu20(t.o))

cpdef Tensor relu6(Tensor t):
    return Tensor.wrapc(crelu6(t.o))

cpdef Tensor relu5(Tensor t):
    return Tensor.wrapc(crelu5(t.o))
