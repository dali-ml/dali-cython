cdef extern from "dali/tensor/Solver.h" nogil:
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
        CSGD(vector[CMat[T]]&, T clipval, T regc)
        void step(vector[CMat[T]]&)
        void step(vector[CMat[T]]&, T step_size)
        void reset_caches(vector[CMat[T]]&)

    cdef cppclass CAdaGrad "Solver::AdaGrad" [T]:
        T clipval
        T smooth_eps
        T regc
        T step_size
        CAdaGrad()
        CAdaGrad(T smooth_eps, T clipval, T regc)
        CAdaGrad(vector[CMat[T]]&, T smooth_eps, T clipval, T regc)
        void step(vector[CMat[T]]&)
        void step(vector[CMat[T]]&, T step_size)
        void reset_caches(vector[CMat[T]]&)
        void create_gradient_caches(vector[CMat[T]]&)

    cdef cppclass CRMSProp "Solver::RMSProp" [T]:
        T clipval
        T smooth_eps
        T regc
        T step_size
        T decay_rate
        CRMSProp()
        CRMSProp(T decay_rate, T smooth_eps, T clipval, T regc)
        CRMSProp(vector[CMat[T]]&, T decay_rate, T smooth_eps, T clipval, T regc)
        void step(vector[CMat[T]]&)
        void step(vector[CMat[T]]&, T step_size)
        void reset_caches(vector[CMat[T]]&)
        void create_gradient_caches(vector[CMat[T]]&)

    cdef cppclass CAdaDelta "Solver::AdaDelta" [T]:
        T clipval
        T smooth_eps
        T regc
        T rho
        CAdaDelta()
        CAdaDelta(T rho, T smooth_eps, T clipval, T regc)
        CAdaDelta(vector[CMat[T]]&, T rho, T smooth_eps, T clipval, T regc)
        void step(vector[CMat[T]]&)
        void reset_caches(vector[CMat[T]]&)
        void create_gradient_caches(vector[CMat[T]]&)

    cdef cppclass CAdam "Solver::Adam" [T]:
        T clipval
        T smooth_eps
        T regc
        T b1
        T b2
        unsigned long long epoch
        CAdam()
        CAdam(T b1, T b2, T smooth_eps, T clipval, T regc)
        CAdam(vector[CMat[T]]&, T b1, T b2, T smooth_eps, T clipval, T regc)
        void step(vector[CMat[T]]&)
        void step(vector[CMat[T]]&, T step_size)
        void reset_caches(vector[CMat[T]]&)
        void create_gradient_caches(vector[CMat[T]]&)

cdef class SGD:
    cdef CSGD[dtype] solverinternal

    property step_size:
        def __get__(SGD self):
            return self.solverinternal.step_size

        def __set__(SGD self, float val):
            self.solverinternal.step_size = val

    property clipval:
        def __get__(SGD self):
            return self.solverinternal.clipval

        def __set__(SGD self, float val):
            self.solverinternal.clipval = val

    property regc:
        def __get__(SGD self):
            return self.solverinternal.regc

        def __set__(SGD self, float val):
            self.solverinternal.regc = val

    property smooth_eps:
        def __get__(SGD self):
            return self.solverinternal.smooth_eps

        def __set__(SGD self, float val):
            self.solverinternal.smooth_eps = val

    def __cinit__(SGD self, params = None, float clipval = 5.0, float regc = 0.0, float step_size = 0.01):
        cdef vector[CMat[dtype]] c_params
        if params is not None:
            for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
        self.solverinternal = CSGD[dtype](c_params, clipval, regc)
        self.solverinternal.step_size = step_size

    def reset_caches(SGD self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        self.solverinternal.reset_caches(params_vec)

    def step(SGD self, list params, step_size = None):
        cdef dtype cstep_size = self.solverinternal.step_size
        if step_size is not None:
            cstep_size = step_size
        cdef vector[CMat[dtype]] c_params
        for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
        with nogil:
            self.solverinternal.step(c_params, cstep_size)

cdef class AdaGrad:
    cdef CAdaGrad[dtype] solverinternal

    property step_size:
        def __get__(AdaGrad self):
            return self.solverinternal.step_size

        def __set__(AdaGrad self, float val):
            self.solverinternal.step_size = val

    property clipval:
        def __get__(AdaGrad self):
            return self.solverinternal.clipval

        def __set__(AdaGrad self, float val):
            self.solverinternal.clipval = val

    property regc:
        def __get__(AdaGrad self):
            return self.solverinternal.regc

        def __set__(AdaGrad self, float val):
            self.solverinternal.regc = val

    property smooth_eps:
        def __get__(AdaGrad self):
            return self.solverinternal.smooth_eps

        def __set__(AdaGrad self, float val):
            self.solverinternal.smooth_eps = val

    def __cinit__(AdaGrad self, params = None, float eps = 1e-6, float clipval = 5.0, float regc = 0.0, float step_size = 0.01):
        cdef vector[CMat[dtype]] c_params
        if params is not None:
            for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
            self.solverinternal = CAdaGrad[dtype](c_params, eps, clipval, regc)
        else:
            self.solverinternal = CAdaGrad[dtype](eps, clipval, regc)
        self.solverinternal.step_size = step_size

    def reset_caches(AdaGrad self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        self.solverinternal.reset_caches(params_vec)

    def create_gradient_caches(AdaGrad self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        self.solverinternal.create_gradient_caches(params_vec)

    def step(AdaGrad self, list params, step_size = None):
        cdef dtype cstep_size = self.solverinternal.step_size
        if step_size is not None:
            cstep_size = step_size
        cdef vector[CMat[dtype]] c_params
        for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
        with nogil:
            self.solverinternal.step(c_params, cstep_size)

cdef class RMSProp:
    cdef CRMSProp[dtype] solverinternal

    property step_size:
        def __get__(RMSProp self):
            return self.solverinternal.step_size

        def __set__(RMSProp self, float val):
            self.solverinternal.step_size = val

    property decay_rate:
        def __get__(RMSProp self):
            return self.solverinternal.decay_rate

        def __set__(RMSProp self, float val):
            self.solverinternal.decay_rate = val

    property clipval:
        def __get__(RMSProp self):
            return self.solverinternal.clipval

        def __set__(RMSProp self, float val):
            self.solverinternal.clipval = val

    property regc:
        def __get__(RMSProp self):
            return self.solverinternal.regc

        def __set__(RMSProp self, float val):
            self.solverinternal.regc = val

    property smooth_eps:
        def __get__(RMSProp self):
            return self.solverinternal.smooth_eps

        def __set__(RMSProp self, float val):
            self.solverinternal.smooth_eps = val

    def __cinit__(RMSProp self, params = None, float decay_rate = 0.999, float eps = 1e-6, float clipval = 5.0, float regc = 0.0, float step_size = 0.01):
        cdef vector[CMat[dtype]] c_params
        if params is not None:
            for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
            self.solverinternal = CRMSProp[dtype](c_params, decay_rate, eps, clipval, regc)
        else:
            self.solverinternal = CRMSProp[dtype](decay_rate, eps, clipval, regc)
        self.solverinternal.step_size = step_size

    def reset_caches(RMSProp self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        with nogil:
            self.solverinternal.reset_caches(params_vec)

    def create_gradient_caches(RMSProp self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        self.solverinternal.create_gradient_caches(params_vec)

    def step(RMSProp self, list params, step_size = None):
        cdef dtype cstep_size = self.solverinternal.step_size
        if step_size is not None:
            cstep_size = step_size
        cdef vector[CMat[dtype]] c_params
        for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
        with nogil:
            self.solverinternal.step(c_params, cstep_size)

cdef class AdaDelta:
    cdef CAdaDelta[dtype] solverinternal

    property rho:
        def __get__(AdaDelta self):
            return self.solverinternal.rho

        def __set__(AdaDelta self, float val):
            self.solverinternal.rho = val

    property clipval:
        def __get__(AdaDelta self):
            return self.solverinternal.clipval

        def __set__(AdaDelta self, float val):
            self.solverinternal.clipval = val

    property regc:
        def __get__(AdaDelta self):
            return self.solverinternal.regc

        def __set__(AdaDelta self, float val):
            self.solverinternal.regc = val

    property smooth_eps:
        def __get__(AdaDelta self):
            return self.solverinternal.smooth_eps

        def __set__(AdaDelta self, float val):
            self.solverinternal.smooth_eps = val

    def __cinit__(AdaDelta self, params = None, float rho = 0.95, float eps = 1e-4, float clipval = 5.0, float regc = 0.0, float step_size = 0.01):
        cdef vector[CMat[dtype]] c_params
        if params is not None:
            for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
            self.solverinternal = CAdaDelta[dtype](c_params, rho, eps, clipval, regc)
        else:
            self.solverinternal = CAdaDelta[dtype](rho, eps, clipval, regc)

    def reset_caches(AdaDelta self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        with nogil:
            self.solverinternal.reset_caches(params_vec)

    def create_gradient_caches(AdaDelta self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        with nogil:
            self.solverinternal.create_gradient_caches(params_vec)

    def step(AdaDelta self, list params):
        cdef vector[CMat[dtype]] c_params
        for param in params:
            assert(type(param) is Mat), "Parameters must be of type Mat"
            c_params.push_back((<Mat>param).matinternal)
        with nogil:
            self.solverinternal.step(c_params)

cdef class Adam:
    cdef CAdam[dtype] solverinternal

    property b1:
        def __get__(Adam self):
            return self.solverinternal.b1

        def __set__(Adam self, float val):
            self.solverinternal.b1 = val

    property b2:
        def __get__(Adam self):
            return self.solverinternal.b2

        def __set__(Adam self, float val):
            self.solverinternal.b2 = val

    property epoch:
        def __get__(Adam self):
            return self.solverinternal.epoch

        def __set__(Adam self, unsigned long long val):
            self.solverinternal.epoch = val

    property clipval:
        def __get__(Adam self):
            return self.solverinternal.clipval

        def __set__(Adam self, float val):
            self.solverinternal.clipval = val

    property regc:
        def __get__(Adam self):
            return self.solverinternal.regc

        def __set__(Adam self, float val):
            self.solverinternal.regc = val

    property smooth_eps:
        def __get__(Adam self):
            return self.solverinternal.smooth_eps

        def __set__(Adam self, float val):
            self.solverinternal.smooth_eps = val

    def __cinit__(Adam self, params = None, float b1 = 0.5, float b2 = 1e-6, float eps = 1e-4, float clipval = 5.0, float regc = 0.0, float step_size = 0.01):
        cdef vector[CMat[dtype]] c_params
        if params is not None:
            for param in params:
                assert(type(param) is Mat), "Parameters must be of type Mat"
                c_params.push_back((<Mat>param).matinternal)
            self.solverinternal = CAdam[dtype](c_params, b1, b2, eps, clipval, regc)
        else:
            self.solverinternal = CAdam[dtype](b1, b2, eps, clipval, regc)

    def reset_caches(Adam self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        with nogil:
            self.solverinternal.reset_caches(params_vec)

    def create_gradient_caches(Adam self, list params):
        cdef vector[CMat[dtype]] params_vec = list_mat_to_vector_mat(params)
        with nogil:
            self.solverinternal.create_gradient_caches(params_vec)

    def step(Adam self, list params, float step_size = 0.0002):
        cdef vector[CMat[dtype]] c_params
        for param in params:
            assert(type(param) is Mat), "Parameters must be of type Mat"
            c_params.push_back((<Mat>param).matinternal)
        with nogil:
            self.solverinternal.step(c_params, step_size)
