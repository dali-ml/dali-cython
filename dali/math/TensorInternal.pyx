cdef extern from "dali/math/TensorInternal.h":
    cdef cppclass TensorInternal [T]:
        T* data() const
        TensorInternal(const TensorInternal[T]& other)
