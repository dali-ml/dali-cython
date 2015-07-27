cdef extern from "dali/math/TensorInternal.h":
    cdef cppclass TensorInternal [T]:
        T* data() const
