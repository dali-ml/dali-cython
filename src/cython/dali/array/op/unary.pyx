from ..array cimport AssignableArray, Array, CAssignableArray, CArray

cpdef AssignableArray sigmoid(Array a):
    return AssignableArray.wrapc(c_sigmoid(a.o))
