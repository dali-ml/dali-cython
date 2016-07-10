from dali.array.array cimport *

cdef extern from "dali/array/op/unary.h" namespace "op" nogil:
    CAssignableArray c_sigmoid "op::sigmoid"(const CArray& x)

cpdef AssignableArray sigmoid(Array a)
