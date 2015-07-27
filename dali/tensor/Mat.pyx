ctypedef unsigned int dim_t;
from cython.operator cimport dereference as deref
from libcpp11.stringstream cimport stringstream

cdef extern from "dali/tensor/Mat.h":
    cdef cppclass CMat "Mat" [T]:
        shared_ptr[string] name
        CMat()
        CMat(dim_t, dim_t)
        vector[dim_t] dims() const
        void npy_load(string fname)
        void npy_save(string fname, string mode)
        int id() const
        unsigned int number_of_elements() const
        dim_t dims(int idx)
        CMat[T] operator_plus   "operator+"(CMat[T] other) except +
        CMat[T] operator_plus   "operator+"(T other) except +
        CMat[T] operator_minus  "operator-"(CMat[T] other) except +
        CMat[T] operator_minus  "operator-"(T other) except +
        CMat[T] operator_times  "operator*"(CMat[T] other) except +
        CMat[T] operator_times  "operator*"(T other) except +
        CMat[T] operator_divide "operator/"(CMat[T] other) except +
        CMat[T] operator_divide "operator/"(T other) except +
        CMat[T] operator_pow    "operator^"(T other) except +
        CMat[T] operator_pow_mat"operator^"(CMat[T] other) except +

        CMat[T] sum()                  except +
        CMat[T] mean()                 except +
        CMat[T] L2_norm()              except +

        CMat[T] sigmoid()              except +
        CMat[T] tanh()                 except +
        CMat[T] relu()                 except +
        CMat[T] absolute_value "abs"() except +
        CMat[T] square()               except +
        CMat[T] exp()                  except +

        void clear_grad()
        void grad() except +
        void set_name(string& name)
        void print_me "print" (stringstream& stream)
        CMat[T] dot(CMat[T] other) except+

cdef extern from "dali/tensor/matrix_initializations.h":
    cdef cppclass matrix_initializations [T]:
        @staticmethod
        CMat[T] uniform(T low, T high, int rows, int cols)
        @staticmethod
        CMat[T] gaussian(T mean, T std, int rows, int cols)
        @staticmethod
        CMat[T] eye(T diag, int width)
        @staticmethod
        CMat[T] bernoulli(T prob, int rows, int cols)
        @staticmethod
        CMat[T] bernoulli_normalized(T prob, int rows, int cols)
        @staticmethod
        CMat[T] empty(int rows, int cols)
        @staticmethod
        CMat[T] ones(int rows, int cols)

cdef class Mat:
    cdef CMat[dtype] matinternal
    def __cinit__(Mat self, int n, int d):
        assert(n > -1 and d > -1), "Only positive dimensions may be used."
        self.matinternal = CMat[dtype](n, d)

    def dims(Mat self):
        return tuple(self.matinternal.dims())

    property shape:
        def __get__(self):
            return tuple(self.matinternal.dims())

    def npy_save(Mat self, str fname, str mode = "w"):
        cdef string fname_norm = normalize_s(fname)
        cdef string mode_norm = normalize_s(mode)
        self.matinternal.npy_save(fname_norm, mode_norm)

    def npy_load(Mat self, str fname):
        cdef string fname_norm = normalize_s(fname)
        self.matinternal.npy_load(fname_norm)

    def clear_grad(self):
        self.matinternal.clear_grad()

    def grad(self):
        self.matinternal.grad()

    property name:
        def __get__(self):
            cdef string name
            if self.matinternal.name != NULL:
                name = deref(self.matinternal.name)
                return name.decode("utf-8")
            return None
        def __set__(self, str newname):
            self.matinternal.set_name(newname.encode("utf-8"))

    def __add__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal.operator_plus( (<Mat>other).matinternal )
        elif type(other) is float or type(other) is int:
            output.matinternal = self.matinternal.operator_plus( (<dtype>other) )
        else:
            raise TypeError("Mat can only be added to float, int, or Mat.")
        return output

    def __repr__(Mat self):
        cdef stringstream ss
        self.matinternal.print_me(ss)
        return ss.to_string().decode("utf-8")

    def __str__(Mat self):
        cdef string name
        if self.matinternal.name != NULL:
            name = deref(self.matinternal.name)
            return "<Mat name=\"%s\" n=%d, d=%d>" % (name.decode("utf-8"), self.matinternal.dims(0), self.matinternal.dims(1))
        return "<Mat n=%d, d=%d>" % (self.matinternal.dims(0), self.matinternal.dims(1))

    def __sub__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal.operator_minus((<Mat>other).matinternal)
        elif type(other) is float or type(other) is int:
            output.matinternal = self.matinternal.operator_minus((<dtype>other))
        else:
            raise TypeError("Mat can only be substracted by float, int, or Mat.")
        return output

    def __pow__(Mat self, other, modulo):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal.operator_pow_mat((<Mat>other).matinternal)
        elif type(other) is float or type(other) is int:
            output.matinternal = self.matinternal.operator_pow((<dtype>other))
        else:
            raise TypeError("Mat can only be raised to a power by float, int, or Mat.")
        return output

    def __mul__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal.operator_times((<Mat>other).matinternal)
        elif type(other) is float or type(other) is int:
            output.matinternal = self.matinternal.operator_times((<dtype>other))
        else:
            raise TypeError("Mat can only be multiplied by float, int or Mat.")
        return output

    def __truediv__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal.operator_divide((<Mat>other).matinternal)
        elif type(other) is float or type(other) is int:
            output.matinternal = self.matinternal.operator_divide((<dtype>other))
        else:
            raise TypeError("Mat can only be divided by float, int or Mat.")
        return output

    def dot(Mat self, Mat other):
        return WrapMat(self.matinternal.dot(other.matinternal))

    def sum(Mat self):
        return WrapMat(self.matinternal.sum())

    def mean(Mat self):
        return WrapMat(self.matinternal.mean())

    def L2_norm(Mat self):
        return WrapMat(self.matinternal.L2_norm())

    def sigmoid(Mat self):
        return WrapMat(self.matinternal.sigmoid())

    def tanh(Mat self):
        return WrapMat(self.matinternal.tanh())

    def relu(Mat self):
        return WrapMat(self.matinternal.relu())

    def __abs__(Mat self):
        return WrapMat(self.matinternal.absolute_value())

    def square(Mat self):
        return WrapMat(self.matinternal.square())

    def exp(Mat self):
        return WrapMat(self.matinternal.exp())

    @staticmethod
    def eye(rows, float diag = 1.0):
        return WrapMat(matrix_initializations[dtype].eye(diag, rows))

    @staticmethod
    def empty(shape):
        cdef Mat output = Mat(0,0)
        if type(shape) == list or type(shape) == tuple:
            output.matinternal = matrix_initializations[dtype].empty(shape[0], shape[1])
        elif type(shape) == int:
            output.matinternal = matrix_initializations[dtype].empty(shape, 1)
        else:
            raise TypeError("shape must be of type int, list, or tuple.")
        return output

    @staticmethod
    def ones(shape):
        cdef Mat output = Mat(0,0)
        if type(shape) == list or type(shape) == tuple:
            output.matinternal = matrix_initializations[dtype].ones(shape[0], shape[1])
        elif type(shape) == int:
            output.matinternal = matrix_initializations[dtype].ones(shape, 1)
        else:
            raise TypeError("shape must be of type int, list, or tuple.")
        return output

    @staticmethod
    def zeros(shape):
        if type(shape) == list or type(shape) == tuple:
            return Mat(shape[0], shape[1])
        elif type(shape) == int:
            return Mat(shape, 1)
        else:
            raise TypeError("shape must be of type int, list, or tuple.")

cdef inline vector[CMat[dtype]] list_mat_to_vector_mat(list mats):
    cdef vector[CMat[dtype]] mats_vec
    mats.reserve(len(mats))
    for mat in mats:
        mats_vec.push_back((<Mat>mat).matinternal)
    return mats_vec

cdef inline Mat WrapMat(const CMat[dtype]& internal):
    cdef Mat output = Mat(0,0)
    output.matinternal = internal
    return output
