from dali.tensor.tensor          cimport *
from dali.array.array            cimport *
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/solver.h":
    cdef cppclass CSGD "solver::SGD":
        double clip_abs
        double clip_norm
        double smooth_eps
        double regc
        double step_size
        # default parameters look like overloaded
        # functions to cython:
        CSGD(double step_size, double clip_norm, double regc)
        CSGD()
        CSGD(vector[CTensor]&, double step_size, double clip_norm, double regc)
        void step(vector[CTensor]&)
        void step_double "step" (vector[CTensor]&, double step_size)
        void reset_caches(vector[CTensor]&)

    cdef cppclass CAdaGrad "solver::AdaGrad":
        double clip_abs
        double clip_norm
        double smooth_eps
        double regc
        double step_size
        CAdaGrad()
        CAdaGrad(double step_size, double smooth_eps, double clip_norm, double regc)
        CAdaGrad(vector[CTensor]&, double step_size, double smooth_eps, double clip_norm, double regc)
        void step(vector[CTensor]&) except +
        void step_double "step" (vector[CTensor]&, double step_size) except +
        void reset_caches(vector[CTensor]&) except +
        void create_gradient_caches(vector[CTensor]&)

    cdef cppclass CRMSPropMomentum "solver::RMSPropMomentum":
        double clip_abs
        double clip_norm
        double smooth_eps
        double regc
        double step_size
        double decay_rate
        double momentum
        CRMSPropMomentum()
        CRMSPropMomentum(
            const double& step_size,#=1e-4
            const double& decay_rate,#=0.95
            const double& momentum,#=0.9
            const double& smooth_eps,#=1e-4
            const double& clip_norm,#=100.0
            const double& regc,#=0.0
        )
        CRMSPropMomentum(
            vector[CTensor]&,
            const double& step_size,#=1e-4
            const double& decay_rate,#=0.95
            const double& momentum,#=0.9
            const double& smooth_eps,#=1e-4
            const double& clip_norm,#=100.0
            const double& regc,#=0.0
        )
        void step(vector[CTensor]&) except +
        void step_double "step"(vector[CTensor]&, double step_size) except +
        void reset_caches(vector[CTensor]&) except +
        void create_gradient_caches(vector[CTensor]&)

    cdef cppclass CRMSProp "solver::RMSProp":
        double clip_abs
        double clip_norm
        double smooth_eps
        double regc
        double step_size
        double decay_rate
        CRMSProp()
        CRMSProp(double step_size, double decay_rate, double smooth_eps, double clip_norm, double regc)
        CRMSProp(vector[CTensor]&, double step_size, double decay_rate, double smooth_eps, double clip_norm, double regc)
        void step(vector[CTensor]&) except +
        void step_double "step" (vector[CTensor]&, double step_size) except +
        void reset_caches(vector[CTensor]&) except +
        void create_gradient_caches(vector[CTensor]&)

    cdef cppclass CAdaDelta "solver::AdaDelta":
        double clip_abs
        double clip_norm
        double smooth_eps
        double regc
        double rho
        CAdaDelta()
        CAdaDelta(double rho, double smooth_eps, double clip_norm, double regc)
        CAdaDelta(vector[CTensor]&, double rho, double smooth_eps, double clip_norm, double regc)
        void step(vector[CTensor]&) except +
        void reset_caches(vector[CTensor]&) except +
        void create_gradient_caches(vector[CTensor]&)

    cdef cppclass CAdam "solver::Adam":
        double b1
        double b2
        double step_size
        double clip_abs
        double clip_norm
        double smooth_eps
        double regc

        unsigned long long epoch
        CAdam()
        CAdam(double step_size, double b1, double b2, double smooth_eps, double clip_norm, double regc)
        CAdam(vector[CTensor]&, double step_size, double b1, double b2, double smooth_eps, double clip_norm, double regc)
        void step(vector[CTensor]&) except +
        void step_double "step" (vector[CTensor]&, double step_size) except +
        void reset_caches(vector[CTensor]&) except +
        void create_gradient_caches(vector[CTensor]&)

cdef class SGD:
    cdef CSGD o

    @staticmethod
    cdef SGD wrapc(CSGD o)

cdef class Adam:
    cdef CAdam o

    @staticmethod
    cdef Adam wrapc(CAdam o)

cdef class RMSPropMomentum:
    cdef CRMSPropMomentum o

    @staticmethod
    cdef RMSPropMomentum wrapc(CRMSPropMomentum o)

cdef class RMSProp:
    cdef CRMSProp o

    @staticmethod
    cdef RMSProp wrapc(CRMSProp o)

cdef class AdaDelta:
    cdef CAdaDelta o

    @staticmethod
    cdef AdaDelta wrapc(CAdaDelta o)

cdef class AdaGrad:
    cdef CAdaGrad o

    @staticmethod
    cdef AdaGrad wrapc(CAdaGrad o)
