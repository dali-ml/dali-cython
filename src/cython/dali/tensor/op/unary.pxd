from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/unary.h" namespace "tensor_ops":
    CTensor ctanh "tensor_ops::tanh" (const CTensor&)
    CTensor csoftplus "tensor_ops::softplus" (const CTensor&)
    CTensor cabs "tensor_ops::abs" (const CTensor&)
    CTensor clog "tensor_ops::log" (const CTensor&)
    # relu: relu(x) = max(x, 0);
    CTensor crelu "tensor_ops::relu" (const CTensor&)
    # clipped relu: relu(x ; clip) = max( min(clip, x), 0 );
    CTensor crelu "tensor_ops::relu" (const CTensor&, const double& upper_bound);
    CTensor cexp "tensor_ops::exp" (const CTensor&)
    CTensor csigmoid "tensor_ops::sigmoid" (const CTensor&)
    CTensor celtinv "tensor_ops::eltinv" (const CTensor&)
    CTensor csqrt "tensor_ops::sqrt" (const CTensor&)
    CTensor csquare "tensor_ops::square" (const CTensor&)
    CTensor ccube "tensor_ops::cube" (const CTensor&)
    CTensor crsqrt "tensor_ops::rsqrt" (const CTensor&)
    CTensor celtmax "tensor_ops::eltmax" (const CTensor&, const double& lower_bound);
    CTensor celtmin "tensor_ops::eltmin" (const CTensor&, const double& upper_bound);
    CTensor csteep_sigmoid "tensor_ops::steep_sigmoid" (const CTensor&, const double& agressiveness);

    CTensor crelu100 "tensor_ops::relu100" (const CTensor&)
    CTensor crelu20 "tensor_ops::relu20" (const CTensor&)
    CTensor crelu6 "tensor_ops::relu6" (const CTensor&)
    CTensor crelu5 "tensor_ops::relu5" (const CTensor&)


cpdef Tensor tanh(Tensor t)
cpdef Tensor softplus(Tensor t)
cpdef Tensor abs(Tensor t)
cpdef Tensor log(Tensor t)

# clipped relu: relu(x ; clip) = max( min(clip, x), 0 );
cpdef Tensor relu(Tensor t, upper_bound=?)
cpdef Tensor exp(Tensor t)
cpdef Tensor sigmoid(Tensor t)
cpdef Tensor eltinv(Tensor t)
cpdef Tensor sqrt(Tensor t)
cpdef Tensor square(Tensor t)
cpdef Tensor cube(Tensor t)
cpdef Tensor rsqrt(Tensor t)
cpdef Tensor eltmax(Tensor t, double lower_bound)
cpdef Tensor eltmin(Tensor t, double upper_bound)
cpdef Tensor steep_sigmoid(Tensor t, double agressiveness=?)
cpdef Tensor relu100(Tensor t)
cpdef Tensor relu20(Tensor t)
cpdef Tensor relu6(Tensor t)
cpdef Tensor relu5(Tensor t)
