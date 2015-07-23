cdef extern from "dali/tensor/Solver.h":
    cdef cppclass CSGD "Solver::SGD" [T]:
        T clipval
        T smooth_eps
        T regc
        T step_size
        # default parameters look like overloaded
        # functions to cython:
        CSGD(T clipval, T smooth_eps, T regc)
        CSGD(T clipval, T regc)
        CSGD(T clipval)
        CSGD()
        CSGD(vector[CMat[T]]&)
        CSGD(vector[CMat[T]]&, T clipval)
        CSGD(vector[CMat[T]]&, T clipval, T regc)
        void step(vector[CMat[T]]&)
        void step(vector[CMat[T]]&, T step_size)
        void reset_caches(vector[CMat[T]]&)

cdef class SGD:
    cdef CSGD["double"] solverinternal

    property step_size:
        def __get__(self):
            return self.solverinternal.step_size

        def __set__(self, float val):
            self.solverinternal.step_size = val

    property clipval:
        def __get__(self):
            return self.solverinternal.clipval

        def __set__(self, float val):
            self.solverinternal.clipval = val

    property regc:
        def __get__(self):
            return self.solverinternal.regc

        def __set__(self, float val):
            self.solverinternal.regc = val

    property smooth_eps:
        def __get__(self):
            return self.solverinternal.smooth_eps

        def __set__(self, float val):
            self.solverinternal.smooth_eps = val

    def __cinit__(self, params = None, float clipval = 5.0, float regc = 0.0):
        cdef vector[CMat["double"]] c_params
        if params is not None:
            for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
            self.solverinternal = CSGD["double"](c_params, clipval, regc)
        else:
            self.solverinternal = CSGD["double"](clipval, regc)

    def step(self, Mat[:] params):
        cdef vector[CMat["double"]] c_params
        for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
        self.solverinternal.step(c_params)
