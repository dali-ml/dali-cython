from dali.array.array            cimport *
from third_party.libcpp11.vector cimport vector
from libcpp.string cimport string

cdef extern from "dali/array/op/reshape.h" namespace "op":
    CAssignableArray c_concatenate "op::concatenate" (const vector[CArray]& tensors, int axis) except +
    CAssignableArray c_hstack "op::hstack" (const vector[CArray]& tensors) except +
    CAssignableArray c_vstack "op::vstack" (const vector[CArray]& tensors) except +
    CAssignableArray c_gather "op::gather"(const CArray& params, const CArray& indices) except +
    CAssignableArray c_take_from_rows "op::take_from_rows"(const CArray& source, const CArray& indices) except +

cpdef AssignableArray concatenate(arrays, int axis)
cpdef AssignableArray hstack(arrays)
cpdef AssignableArray vstack(arrays)
cpdef AssignableArray gather(Array params, Array indices)
cpdef AssignableArray take_from_rows(Array source, Array indices)
