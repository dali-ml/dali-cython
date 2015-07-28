cdef extern from "dali/layers/LSTM.h":
	cdef cppclass CStackedLSTM "StackedLSTM" [T]:
		pass
