cdef extern from "dali/tensor/matrix_initializations.h" nogil:
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

class random:
    @staticmethod
    def uniform(float low = 0.0, float high = 1.0, size=None):
        cdef Mat output = Mat(0,0)
        if type(size) == list or type(size) == tuple:
            output.matinternal = matrix_initializations[dtype].uniform(low, high, size[0], size[1])
        elif type(size) == int:
            output.matinternal = matrix_initializations[dtype].uniform(low, high, size, 1)
        else:
            raise ValueError("size must be of type int, tuple, or list")
        return output

    @staticmethod
    def normal(float loc=0.0, float scale=1.0, size=None):
        cdef Mat output = Mat(0,0)
        if type(size) == list or type(size) == tuple:
            output.matinternal = matrix_initializations[dtype].gaussian(loc, scale, size[0], size[1])
        elif type(size) == int:
            output.matinternal = matrix_initializations[dtype].gaussian(loc, scale, size, 1)
        else:
            raise ValueError("size must be of type int, tuple, or list")
        return output

    @staticmethod
    def standard_normal(size=None):
        cdef Mat output = Mat(0,0)
        if type(size) == list or type(size) == tuple:
            output.matinternal = matrix_initializations[dtype].gaussian(0.0, 1.0, size[0], size[1])
        elif type(size) == int:
            output.matinternal = matrix_initializations[dtype].gaussian(0.0, 1.0, size, 1)
        else:
            raise ValueError("size must be of type int, tuple, or list")
        return output

    @staticmethod
    def bernoulli(float prob, size=None):
        cdef Mat output = Mat(0,0)
        if type(size) == list or type(size) == tuple:
            output.matinternal = matrix_initializations[dtype].bernoulli(prob, size[0], size[1])
        elif type(size) == int:
            output.matinternal = matrix_initializations[dtype].bernoulli(prob, size, 1)
        else:
            raise ValueError("size must be of type int, tuple, or list")
        return output

    @staticmethod
    def bernoulli_normalized(float prob, size=None):
        cdef Mat output = Mat(0,0)
        if type(size) == list or type(size) == tuple:
            output.matinternal = matrix_initializations[dtype].bernoulli_normalized(prob, size[0], size[1])
        elif type(size) == int:
            output.matinternal = matrix_initializations[dtype].bernoulli_normalized(prob, size, 1)
        else:
            raise ValueError("size must be of type int, tuple, or list")
        return output
