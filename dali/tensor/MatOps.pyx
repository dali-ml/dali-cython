cdef extern from "dali/tensor/MatOps.h":
    cdef cppclass CMatOps "MatOps" [T]:
        ### OTHER ###
        @staticmethod
        CMat[T] fill(CMat[T] to_fill, T filler)

        @staticmethod
        CMat[T] consider_constant(CMat[T])

        @staticmethod
        bint equals(CMat[T], CMat[T])

        @staticmethod
        bint allclose(CMat[T], CMat[T], T tol)

        @staticmethod
        bint grad_allclose(CMat[T], CMat[T], T tol)

        @staticmethod
        vector[int] argsort(CMat[T])

        @staticmethod
        vector[size_t] argsort(const vector[CMat[T]]&)

        @staticmethod
        int argmax(const CMat[T]&) except +

        @staticmethod
        int argmin(const CMat[T]&) except +

        @staticmethod
        vector[int] argmax_axis "argmax" (const CMat[T]&, int dimension)

        @staticmethod
        vector[int] argmin_axis "argmin" (const CMat[T]&, int dimension)

        @staticmethod
        int argmax_slice(const CMat[T]&, int lower, int upper)

        @staticmethod
        int argmin_slice(const CMat[T]&, int lower, int upper)

        @staticmethod
        void copy(CMat[T]* dest, const CMat[T]& source)

        @staticmethod
        void copy_grad(CMat[T]* dest, const CMat[T]& source)

        ### REDUCERS ###
        @staticmethod
        CMat[T] L2_norm(CMat[T])

        @staticmethod
        CMat[T] sum(CMat[T])

        @staticmethod
        CMat[T] mean(CMat[T])

        ### RESHAPING ###

        @staticmethod
        CMat[T] hstack(CMat[T], CMat[T])

        @staticmethod
        CMat[T] hstack(const vector[CMat[T]]&)

        @staticmethod
        CMat[T] vstack(CMat[T], CMat[T])

        @staticmethod
        CMat[T] vstack(const vector[CMat[T]]&)

        @staticmethod
        CMat[T] transpose(CMat[T])

        @staticmethod
        CMat[T] rows_pluck(CMat[T], CMat[int])

        # requires Indexing::Index
        # @staticmethod
        # CMat[T] rows_pluck

        # @staticmethod
        # CMat[T] rows_cols_pluck

        @staticmethod
        CMat[T] row_pluck(CMat[T], int)

        @staticmethod
        CMat[T] col_pluck(CMat[T], int)

        @staticmethod
        void resize(CMat[T]& mat, unsigned int rows, unsigned int cols)

        ### SOLVER_UPDATES ###
        @staticmethod
        void clip_and_regularize(CMat[T] param, T clipval, T regc)

        @staticmethod
        void sgd_update(CMat[T] param, T step_size)

        # requires TensorInternal
        # @staticmethod
        # void adagrad_update(CMat[T] param, T step_size)

        # @staticmethod
        # void rmsprop_update

        # @staticmethod
        # void adadelta_update

        # @staticmethod
        # void adam_update

        ### ELEMWISE ###

        @staticmethod
        CMat[T] add(CMat[T], T)

        @staticmethod
        CMat[T] sub_broadcast_reversed(CMat[T], T)

        @staticmethod
        CMat[T] eltmul(CMat[T], T)

        @staticmethod
        CMat[T] eltdivide(CMat[T], T)

        @staticmethod
        CMat[T] max(CMat[T], T)

        @staticmethod
        CMat[T] square(CMat[T])

        @staticmethod
        CMat[T] log(CMat[T])

        @staticmethod
        CMat[T] exp(CMat[T])

        @staticmethod
        CMat[T] sigmoid(CMat[T])

        @staticmethod
        CMat[T] steep_sigmoid(CMat[T], T aggressiveness)

        @staticmethod
        CMat[T] tanh(CMat[T])

        @staticmethod
        CMat[T] relu(CMat[T])

        @staticmethod
        CMat[T] abs(CMat[T])

        @staticmethod
        CMat[T] pow(CMat[T], T power)

        @staticmethod
        CMat[T] sqrt(CMat[T])

        @staticmethod
        CMat[T] elt_inv(CMat[T])

        ### DROPOUT ###
        CMat[T] dropout(CMat[T], T drop_prob)
        CMat[T] dropout_normalized(CMat[T], T drop_prob)
        CMat[T] fast_dropout(CMat[T])

        vector[CMat[T]] dropout(const vector[CMat[T]]&, T drop_prob)
        vector[CMat[T]] dropout_normalized(const vector[CMat[T]]&, T drop_prob)
        vector[CMat[T]] fast_dropout(const vector[CMat[T]]&)

cdef class MatOps:
    @staticmethod
    def fill(Mat mat, float filler):
        cdef Mat output = Mat(0,0)
        output.matinternal = CMatOps[dtype].fill(mat.matinternal, filler)
        return output

    @staticmethod
    def consider_constant(Mat mat):
        cdef Mat output = Mat(0,0)
        output.matinternal = CMatOps[dtype].consider_constant(mat.matinternal)
        return output

    @staticmethod
    def equals(Mat a, Mat b):
        return CMatOps[dtype].equals(a.matinternal, b.matinternal)

    @staticmethod
    def allclose(Mat a, Mat b, float tol = 1e-6):
        return CMatOps[dtype].allclose(a.matinternal, b.matinternal, tol)

    @staticmethod
    def grad_allclose(Mat a, Mat b, float tol = 1e-6):
        return CMatOps[dtype].grad_allclose(a.matinternal, b.matinternal, tol)

    @staticmethod
    def argsort(Mat mat):
        return CMatOps[dtype].argsort(mat.matinternal)

    @staticmethod
    def argmax(Mat mat, axis=None):
        if axis is not None:
            return CMatOps[dtype].argmax_axis(mat.matinternal, int(axis))
        return CMatOps[dtype].argmax(mat.matinternal)

    @staticmethod
    def argmin(Mat mat, axis=None):
        if axis is not None:
            return CMatOps[dtype].argmin_axis(mat.matinternal, int(axis))
        return CMatOps[dtype].argmin(mat.matinternal)

    @staticmethod
    def argmin_slice(Mat mat, int lower, int upper):
        return CMatOps[dtype].argmin_slice(mat.matinternal, lower, upper)

    @staticmethod
    def argmax_slice(Mat mat, int lower, int upper):
        return CMatOps[dtype].argmax_slice(mat.matinternal, lower, upper)
