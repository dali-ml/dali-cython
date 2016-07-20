from dali.array.array   cimport *
from dali.tensor.tensor cimport *

from dali.array.op  cimport binary as A
from dali.tensor.op cimport binary as T

from dali.array.op  cimport unary_scalar as scalar_A
from dali.tensor.op cimport unary_scalar as scalar_T

cdef bint is_scalar(obj):
    return isinstance(obj, (float, int))

cpdef add(left, right):
    if not isinstance(left, (Array, Tensor, int, float)):
        left = Array(left)

    if not isinstance(right, (Array, Tensor, int, float)):
        right = Array(right)

    if type(left) == Tensor or type(right) == Tensor:
        if is_scalar(right):
            return scalar_T.scalar_add(left, right)
        elif is_scalar(left):
            return scalar_T.scalar_radd(left, right)
        else:
            return T.add(ensure_tensor(left), ensure_tensor(right))
    else:
        if is_scalar(right):
            return scalar_A.scalar_add(ensure_array(left), right)
        elif is_scalar(left):
            return scalar_A.scalar_radd(left, ensure_array(right))
        else:
            return A.add(ensure_array(left), ensure_array(right))
