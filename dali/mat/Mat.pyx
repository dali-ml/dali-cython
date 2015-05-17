ctypedef unsigned int dim_t;

cdef extern from "dali/mat/Mat.h":
    cdef cppclass CMat "Mat" [T]:
        CMat()
        CMat(dim_t, dim_t)
        CMat(dim_t, dim_t, bool)
        vector[dim_t] dims() const
        void npy_load(string fname)
        void npy_save(string fname, string mode)
        int id() const
        unsigned int number_of_elements() const
        dim_t dims(int idx)
        CMat[T] operator+(CMat[T] other) except +
        CMat[T] operator+(T other) except +
        CMat[T] operator-(CMat[T] other) except +
        CMat[T] operator-(T other) except +
        CMat[T] operator*(CMat[T] other) except +
        CMat[T] operator*(T other) except +
        CMat[T] operator/(CMat[T] other) except +
        CMat[T] operator/(T other) except +

cdef class Mat:
    cdef CMat["double"] matinternal
    def __cinit__(Mat self, int n, int d, bint fill_zeros=True):
        assert(n > -1 and d > -1), "Only positive dimensions may be used."
        self.matinternal = CMat["double"](n, d, fill_zeros)

    def dims(Mat self):
        return list(self.matinternal.dims())

    def npy_save(Mat self, str fname, str mode = "w"):
        cdef string fname_norm = normalize_s(fname)
        cdef string mode_norm = normalize_s(mode)
        self.matinternal.npy_save(fname_norm, mode_norm)

    def npy_load(Mat self, str fname):
        cdef string fname_norm = normalize_s(fname)
        self.matinternal.npy_load(fname_norm)

    def __add__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal + (<Mat>other).matinternal
        elif type(other) is float:
            output.matinternal = self.matinternal + (<double>other)
        else:
            raise TypeError("Mat can only be added to float or Mat.")
        return output

    def __sub__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal - (<Mat>other).matinternal
        elif type(other) is float:
            output.matinternal = self.matinternal - (<double>other)
        else:
            raise TypeError("Mat can only be substracted by float or Mat.")
        return output

    def __mul__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal * (<Mat>other).matinternal
        elif type(other) is float:
            output.matinternal = self.matinternal * (<double>other)
        else:
            raise TypeError("Mat can only be multiplied by float or Mat.")
        return output

    def __div__(Mat self, other):
        cdef Mat output = Mat(0,0)
        if type(other) is Mat:
            output.matinternal = self.matinternal / (<Mat>other).matinternal
        elif type(other) is float:
            output.matinternal = self.matinternal / (<double>other)
        else:
            raise TypeError("Mat can only be divided by float or Mat.")
        return output
