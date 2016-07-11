from dali.array.array            cimport *
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/array/op/binary.h" namespace "op" nogil:
    CAssignableArray c_add "op::add" (const CArray& left, const CArray& right)
    CAssignableArray c_add "op::add" (const vector[CArray]& arrays)
    CAssignableArray c_sub "op::sub" (const CArray& left, const CArray& right)
    CAssignableArray c_eltmul "op::eltmul" (const CArray& left, const CArray& right)
    CAssignableArray c_eltdiv "op::eltdiv" (const CArray& left, const CArray& right)
    CAssignableArray c_pow "op::pow" (const CArray& left, const CArray& right)
    CAssignableArray c_equals "op::equals" (const CArray& left, const CArray& right)
    CAssignableArray c_circular_convolution "op::circular_convolution" (const CArray& content, const CArray& shift)
    CAssignableArray c_prelu "op::prelu" (const CArray& x, const CArray& weights)


cpdef AssignableArray add(Array left, Array right)
cpdef AssignableArray add_n(arrays)
cpdef AssignableArray sub(Array left, Array right)
cpdef AssignableArray eltmul(Array left, Array right)
cpdef AssignableArray eltdiv(Array left, Array right)
cpdef AssignableArray pow(Array left, Array right)
cpdef AssignableArray equals(Array left, Array right)
cpdef AssignableArray circular_convolution(Array content, Array shift)
cpdef AssignableArray prelu(Array x, Array weights)
