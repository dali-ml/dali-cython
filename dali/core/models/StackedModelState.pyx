cdef extern from "dali/models/StackedModelState.h":
	cdef cppclass StackedModelState [T]:
		CMat[T] prediction
		CMat[T] memory
		StackedModelState()
