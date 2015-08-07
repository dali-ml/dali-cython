from cython.operator cimport dereference as deref

cdef extern from "dali/models/StackedModel.h":
    cdef cppclass CStackedModel "StackedModel" [T]:
        const bint use_shortcut
        const bint memory_feeds_gates

        int vocabulary_size
        int output_size
        vector[int] hidden_sizes

        CMat[T] embedding

        CStackedLSTM[T] stacked_lstm
        CStackedInputLayer[T] decoder
        vector[CMat[T]] parameters() const
        CStackedModel()
        CStackedModel(int vocabulary_size,
                      int input_size,
                      int output_size,
                      const vector[int]& hidden_sizes,
                      bint use_shortcut,
                      bint memory_feeds_gates)

        CStackedModel(const CStackedModel& other,
                    bint copy_w,
                    bint copy_dw)

        CMat[T] masked_predict_cost(
            CMat[int] data,
            CMat[int] target_data,
            CMat[T] prediction_mask,
            T drop_prob,
            int temporal_offset,
            unsigned int softmax_offset) const

        CMat[T] masked_predict_cost(
            const CBatch[T]& data,
            T drop_prob,
            int temporal_offset,
            unsigned int softmax_offset) const

        # Requires indexing::index
        #vector[int] reconstruct(Indexing::Index, T drop_prob) const

        const bint& get_input_vector_to_decoder "input_vector_to_decoder"() const
        void set_input_vector_to_decoder "input_vector_to_decoder"(
            bint should_input_feed_to_decoder) const

        CStackedModel[T] shallow_copy() const

        # CMat[T] decode(
        #     CMat[T] input_vector,
        #     vector[CLSTMState[T]]&, T drop_prob) const

"""
cdef class StackedModel:
    cdef shared_ptr[CStackedModel[dtype]] layerinternal

    def __cinit__(StackedModel self,
            int vocabulary_size,
            int input_size,
            int output_size,
            list hidden_sizes,
            bint use_shortcut = False,
            bint memory_feeds_gates = False):
        assert(len(hidden_sizes) > 0), "hidden_sizes cannot be empty."
        self.layerinternal = shared_ptr[CStackedModel[dtype]](
            new CStackedModel[dtype](
                vocabulary_size,
                input_size,
                output_size,
                hidden_sizes,
                use_shortcut,
                memory_feeds_gates
            )
        )

    def parameters(StackedModel self):
        params = []
        cdef vector[CMat[dtype]] params_mat = deref(self.layerinternal).parameters()
        for param in params_mat:
            mat = Mat(0,0)
            mat.matinternal = param
            params.append(mat)
        return params

    property embedding:
        def __get__(StackedModel self):
            return WrapMat(deref(self.layerinternal).embedding)

        def __set__(StackedModel self, Mat value):
            deref(self.layerinternal).embedding = value.matinternal
            deref(self.layerinternal).vocabulary_size = deref(self.layerinternal).embedding.dims(0)

    property input_size:
        def __get__(StackedModel self):
            return deref(self.layerinternal).embedding.dims(1)

    property output_size:
        def __get__(StackedModel self):
            return deref(self.layerinternal).output_size

    property vocabulary_size:
        def __get__(StackedModel self):
            return deref(self.layerinternal).vocabulary_size
"""
