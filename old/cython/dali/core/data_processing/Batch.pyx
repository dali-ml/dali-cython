cdef extern from "dali/data_processing/Batch.h":
	cdef cppclass CBatch "Batch" [T]:
		CMat[int] data
		CMat[int] target
		CMat[T]   mask
		vector[int] code_lengths
		int total_codes
		CBatch()
		size_t size() const
		size_t max_length() const
		void insert_example(
			const vector[string]& example,
			const CVocab& vocab,
			size_t example_idx,
			int offset)
		int example_length(const int& idx) const
