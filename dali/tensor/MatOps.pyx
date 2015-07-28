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
        CMat[T] hstack_vec "hstack" (const vector[CMat[T]]&)

        @staticmethod
        CMat[T] vstack(CMat[T], CMat[T])

        @staticmethod
        CMat[T] vstack_vec "vstack"(const vector[CMat[T]]&)

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

        @staticmethod
        CMat[T] dropout(CMat[T], T drop_prob)

        @staticmethod
        CMat[T] dropout_normalized(CMat[T], T drop_prob)

        @staticmethod
        CMat[T] fast_dropout(CMat[T])

        @staticmethod
        vector[CMat[T]] dropout(const vector[CMat[T]]&, T drop_prob)

        @staticmethod
        vector[CMat[T]] dropout_normalized(const vector[CMat[T]]&, T drop_prob)

        @staticmethod
        vector[CMat[T]] fast_dropout(const vector[CMat[T]]&)

        ### COST ###

        @staticmethod
        CMat[T] binary_cross_entropy(CMat[T], T target) except +

        @staticmethod
        CMat[T] sigmoid_binary_cross_entropy(CMat[T], T target) except +

        @staticmethod
        CMat[T] softmax(CMat[T], T temperature) except +

        @staticmethod
        CMat[T] softmax_transpose(CMat[T], T temperature) except +

        @staticmethod
        CMat[T] softmax_no_grad(CMat[T], T temperature) except +

        @staticmethod
        CMat[T] softmax_no_grad_transpose(CMat[T], T temperature) except +

        @staticmethod
        CMat[T] margin_loss(CMat[T], unsigned int answer_idx, T margin) except +

        @staticmethod
        CMat[T] softmax_cross_entropy(CMat[T], unsigned int answer_idx) except +

cdef class MatOps:
    @staticmethod
    def fill(Mat mat, float filler):
        return WrapMat(CMatOps[dtype].fill(mat.matinternal, filler))

    @staticmethod
    def consider_constant(Mat mat):
        return WrapMat(CMatOps[dtype].consider_constant(mat.matinternal))

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

    @staticmethod
    def resize(Mat mat, int rows, int cols):
        assert(rows > -1 and cols > -1), "Can only resize to positive dimensions."
        CMatOps[dtype].resize(mat.matinternal, rows, cols)

    @staticmethod
    def copy(Mat destination, Mat source):
        CMatOps[dtype].copy(&destination.matinternal, source.matinternal)

    @staticmethod
    def copy_grad(Mat destination, Mat source):
        CMatOps[dtype].copy_grad(&destination.matinternal, source.matinternal)

    ### REDUCERS ###
    @staticmethod
    def L2_norm(Mat mat):
        return WrapMat(CMatOps[dtype].L2_norm(mat.matinternal))

    @staticmethod
    def mean(Mat mat):
        return WrapMat(CMatOps[dtype].mean(mat.matinternal))

    @staticmethod
    def sum(Mat mat):
        return WrapMat(CMatOps[dtype].sum(mat.matinternal))

    ### RESHAPING ###
    @staticmethod
    def hstack(Mat left, Mat right):
        return WrapMat(CMatOps[dtype].hstack(left.matinternal, right.matinternal))

    @staticmethod
    def hstack(list mats):
        cdef vector[CMat[dtype]] mats_vec = list_mat_to_vector_mat(mats)
        return WrapMat(CMatOps[dtype].hstack_vec(mats_vec))

    @staticmethod
    def vstack(Mat top, Mat bottom):
        return WrapMat(CMatOps[dtype].vstack(top.matinternal, bottom.matinternal))

    @staticmethod
    def vstack(list mats):
        cdef vector[CMat[dtype]] mats_vec = list_mat_to_vector_mat(mats)
        return WrapMat(CMatOps[dtype].vstack_vec(mats_vec))

    @staticmethod
    def transpose(Mat mat):
        return WrapMat(CMatOps[dtype].transpose(mat.matinternal))

    ### UPDATES ###
    @staticmethod
    def sgd_update(Mat mat, float step_size):
        CMatOps[dtype].sgd_update(mat.matinternal, step_size)

    @staticmethod
    def clip_and_regularize(Mat mat, float clipval = 5.0, float regc = 1e-6):
        CMatOps[dtype].clip_and_regularize(mat.matinternal, clipval, regc)

    ### ELEMWISE ###

    @staticmethod
    def add(Mat mat, float val):
        return WrapMat(CMatOps[dtype].sub_broadcast_reversed(mat.matinternal, val))

    @staticmethod
    def sub_broadcast_reversed(Mat mat, float val):
        return WrapMat(CMatOps[dtype].sub_broadcast_reversed(mat.matinternal, val))

    @staticmethod
    def eltmul(Mat mat, float val):
        return WrapMat(CMatOps[dtype].eltmul(mat.matinternal, val))

    @staticmethod
    def eltdivide(Mat mat, float val):
        return WrapMat(CMatOps[dtype].eltdivide(mat.matinternal, val))

    @staticmethod
    def max(Mat mat, float val):
        return WrapMat(CMatOps[dtype].max(mat.matinternal, val))

    @staticmethod
    def square(Mat mat):
        return WrapMat(CMatOps[dtype].square(mat.matinternal))

    @staticmethod
    def log(Mat mat):
        return WrapMat(CMatOps[dtype].log(mat.matinternal))

    @staticmethod
    def exp(Mat mat):
        return WrapMat(CMatOps[dtype].exp(mat.matinternal))

    @staticmethod
    def sigmoid(Mat mat):
        return WrapMat(CMatOps[dtype].sigmoid(mat.matinternal))

    @staticmethod
    def steep_sigmoid(Mat mat, float aggressiveness):
        return WrapMat(CMatOps[dtype].steep_sigmoid(mat.matinternal, aggressiveness))

    @staticmethod
    def tanh(Mat mat):
        return WrapMat(CMatOps[dtype].tanh(mat.matinternal))

    @staticmethod
    def relu(Mat mat):
        return WrapMat(CMatOps[dtype].relu(mat.matinternal))

    @staticmethod
    def abs(Mat mat):
        return WrapMat(CMatOps[dtype].abs(mat.matinternal))

    @staticmethod
    def pow(Mat mat, float power):
        return WrapMat(CMatOps[dtype].pow(mat.matinternal, power))

    @staticmethod
    def sqrt(Mat mat):
        return WrapMat(CMatOps[dtype].sqrt(mat.matinternal))

    @staticmethod
    def elt_inv(Mat mat):
        return WrapMat(CMatOps[dtype].elt_inv(mat.matinternal))

    # DROPOUT
    @staticmethod
    def dropout(Mat mat, float drop_prob):
        return WrapMat(CMatOps[dtype].dropout(mat.matinternal,drop_prob))

    @staticmethod
    def dropout_normalized(Mat mat, float drop_prob):
        return WrapMat(CMatOps[dtype].dropout_normalized(mat.matinternal,drop_prob))

    @staticmethod
    def fast_dropout(Mat mat):
        return WrapMat(CMatOps[dtype].fast_dropout(mat.matinternal))

    # COST #
    @staticmethod
    def binary_cross_entropy(Mat mat, float target):
        return WrapMat(CMatOps[dtype].binary_cross_entropy(mat.matinternal, target))

    @staticmethod
    def sigmoid_binary_cross_entropy(Mat mat, float target):
        return WrapMat(CMatOps[dtype].sigmoid_binary_cross_entropy(mat.matinternal, target))

    @staticmethod
    def margin_loss(Mat mat, int answer_idx, float margin = 0.1):
        return WrapMat(CMatOps[dtype].margin_loss(mat.matinternal, answer_idx, margin))

    @staticmethod
    def softmax_cross_entropy(Mat mat, int answer_idx):
        return WrapMat(CMatOps[dtype].softmax_cross_entropy(mat.matinternal, answer_idx))

    @staticmethod
    def softmax(Mat mat, float temperature = 1.0, int axis = 0):
        if axis == 0:
            return WrapMat(CMatOps[dtype].softmax(mat.matinternal, temperature))
        elif axis == 1:
            return WrapMat(CMatOps[dtype].softmax_transpose(mat.matinternal, temperature))
        else:
            raise ValueError("axis must be 0 (columnwise) or 1 (rowwise)")

    @staticmethod
    def softmax_no_grad(Mat mat, float temperature = 1.0, int axis = 0):
        if axis == 0:
            return WrapMat(CMatOps[dtype].softmax_no_grad(mat.matinternal, temperature))
        elif axis == 1:
            return WrapMat(CMatOps[dtype].softmax_no_grad_transpose(mat.matinternal, temperature))
        else:
            raise ValueError("axis must be 0 (columnwise) or 1 (rowwise)")
