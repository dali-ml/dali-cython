cdef class SGD:
    @staticmethod
    cdef SGD wrapc(CSGD o):
        ret = SGD()
        ret.o = o
        return ret

    property step_size:
        def __get__(SGD self):
            return self.o.step_size
        def __set__(SGD self, float val):
            self.o.step_size = val
    property clip_abs:
        def __get__(SGD self):
            return self.o.clip_abs
        def __set__(SGD self, float val):
            self.o.clip_abs = val
    property clip_norm:
        def __get__(SGD self):
            return self.o.clip_norm
        def __set__(SGD self, float val):
            self.o.clip_norm = val
    property regc:
        def __get__(SGD self):
            return self.o.regc
        def __set__(SGD self, float val):
            self.o.regc = val
    property smooth_eps:
        def __get__(SGD self):
            return self.o.smooth_eps
        def __set__(SGD self, float val):
            self.o.smooth_eps = val

    def step(SGD self, params, step_size=None):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        if step_size is None:
            self.o.step(params_c)
        else:
            self.o.step_double(params_c, step_size)

    def __cinit__(SGD self,
                  params=None,
                  float step_size=0.01,
                  float clip_abs=0.0,
                  float clip_norm=5.0,
                  float regc=0.0):
        cdef vector[CTensor] params_c
        if params is None:
            self.o = CSGD(step_size, clip_norm, regc)
        else:
            params_c = ensure_tensor_list(params)
            self.o = CSGD(params_c, step_size, clip_norm, regc)
        self.o.clip_abs = clip_abs



cdef class Adam:
    @staticmethod
    cdef Adam wrapc(CAdam o):
        ret = Adam()
        ret.o = o
        return ret

    property b1:
        def __get__(Adam self):
            return self.o.b1
        def __set__(Adam self, float val):
            self.o.b1 = val
    property b2:
        def __get__(Adam self):
            return self.o.b2
        def __set__(Adam self, float val):
            self.o.b2 = val
    property step_size:
        def __get__(Adam self):
            return self.o.step_size
        def __set__(Adam self, float val):
            self.o.step_size = val
    property clip_abs:
        def __get__(Adam self):
            return self.o.clip_abs
        def __set__(Adam self, float val):
            self.o.clip_abs = val
    property clip_norm:
        def __get__(Adam self):
            return self.o.clip_norm
        def __set__(Adam self, float val):
            self.o.clip_norm = val
    property regc:
        def __get__(Adam self):
            return self.o.regc
        def __set__(Adam self, float val):
            self.o.regc = val
    property smooth_eps:
        def __get__(Adam self):
            return self.o.smooth_eps
        def __set__(Adam self, float val):
            self.o.smooth_eps = val

    def step(Adam self, params, step_size=None):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        if step_size is None:
            self.o.step(params_c)
        else:
            self.o.step_double(params_c, step_size)

    def reset_caches(Adam self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.reset_caches(params_c)

    def create_gradient_caches(Adam self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.create_gradient_caches(params_c)

    def __cinit__(Adam self,
                  params=None,
                  float step_size=0.0002,
                  float b1=0.5,
                  float b2=1e-6,
                  float eps=1e-4,
                  float clip_abs=0.0,
                  float clip_norm=5.0,
                  float regc=0.0):
        cdef vector[CTensor] params_c
        if params is None:
            self.o = CAdam(step_size, b1, b2, eps, clip_norm, regc)
        else:
            params_c = ensure_tensor_list(params)
            self.o = CAdam(params_c, step_size, b1, b2, eps, clip_norm, regc)
        self.o.clip_abs = clip_abs

cdef class RMSPropMomentum:
    @staticmethod
    cdef RMSPropMomentum wrapc(CRMSPropMomentum o):
        ret = RMSPropMomentum()
        ret.o = o
        return ret

    property step_size:
        def __get__(RMSPropMomentum self):
            return self.o.step_size
        def __set__(RMSPropMomentum self, float val):
            self.o.step_size = val
    property clip_abs:
        def __get__(RMSPropMomentum self):
            return self.o.clip_abs
        def __set__(RMSPropMomentum self, float val):
            self.o.clip_abs = val
    property clip_norm:
        def __get__(RMSPropMomentum self):
            return self.o.clip_norm
        def __set__(RMSPropMomentum self, float val):
            self.o.clip_norm = val
    property regc:
        def __get__(RMSPropMomentum self):
            return self.o.regc
        def __set__(RMSPropMomentum self, float val):
            self.o.regc = val
    property smooth_eps:
        def __get__(RMSPropMomentum self):
            return self.o.smooth_eps
        def __set__(RMSPropMomentum self, float val):
            self.o.smooth_eps = val
    property decay_rate:
        def __get__(RMSPropMomentum self):
            return self.o.decay_rate
        def __set__(RMSPropMomentum self, float val):
            self.o.decay_rate = val
    property momentum:
        def __get__(RMSPropMomentum self):
            return self.o.momentum
        def __set__(RMSPropMomentum self, float val):
            self.o.momentum = val

    def step(RMSPropMomentum self, params, step_size=None):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        if step_size is None:
            self.o.step(params_c)
        else:
            self.o.step_double(params_c, step_size)

    def reset_caches(RMSPropMomentum self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.reset_caches(params_c)

    def create_gradient_caches(RMSPropMomentum self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.create_gradient_caches(params_c)

    def __cinit__(RMSPropMomentum self,
                  params=None,
                  float step_size=1e-4,
                  float decay_rate=0.95,
                  float momentum=0.9,
                  float smooth_eps=1e-4,
                  float clip_abs=0.0,
                  float clip_norm=100.0,
                  float regc=0.0):
        cdef vector[CTensor] params_c
        if params is None:
            self.o = CRMSPropMomentum(step_size, decay_rate, momentum, smooth_eps, clip_norm, regc)
        else:
            params_c = ensure_tensor_list(params)
            self.o = CRMSPropMomentum(params_c, step_size, decay_rate, momentum, smooth_eps, clip_norm, regc)
        self.o.clip_abs = clip_abs

cdef class RMSProp:
    @staticmethod
    cdef RMSProp wrapc(CRMSProp o):
        ret = RMSProp()
        ret.o = o
        return ret

    property step_size:
        def __get__(RMSProp self):
            return self.o.step_size
        def __set__(RMSProp self, float val):
            self.o.step_size = val
    property clip_abs:
        def __get__(RMSProp self):
            return self.o.clip_abs
        def __set__(RMSProp self, float val):
            self.o.clip_abs = val
    property clip_norm:
        def __get__(RMSProp self):
            return self.o.clip_norm
        def __set__(RMSProp self, float val):
            self.o.clip_norm = val
    property regc:
        def __get__(RMSProp self):
            return self.o.regc
        def __set__(RMSProp self, float val):
            self.o.regc = val
    property smooth_eps:
        def __get__(RMSProp self):
            return self.o.smooth_eps
        def __set__(RMSProp self, float val):
            self.o.smooth_eps = val
    property decay_rate:
        def __get__(RMSProp self):
            return self.o.decay_rate
        def __set__(RMSProp self, float val):
            self.o.decay_rate = val

    def step(RMSProp self, params, step_size=None):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        if step_size is None:
            self.o.step(params_c)
        else:
            self.o.step_double(params_c, step_size)

    def reset_caches(RMSProp self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.reset_caches(params_c)

    def create_gradient_caches(RMSProp self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.create_gradient_caches(params_c)

    def __cinit__(RMSProp self,
                  params=None,
                  float step_size=0.01,
                  float decay_rate=0.999,
                  float smooth_eps=1e-6,
                  float clip_abs=0.0,
                  float clip_norm=5.0,
                  float regc=0.0):
        cdef vector[CTensor] params_c
        if params is None:
            self.o = CRMSProp(step_size, decay_rate, smooth_eps, clip_norm, regc)
        else:
            params_c = ensure_tensor_list(params)
            self.o = CRMSProp(params_c, step_size, decay_rate, smooth_eps, clip_norm, regc)
        self.o.clip_abs = clip_abs

cdef class AdaDelta:
    @staticmethod
    cdef AdaDelta wrapc(CAdaDelta o):
        ret = AdaDelta()
        ret.o = o
        return ret
    property rho:
        def __get__(AdaDelta self):
            return self.o.rho
        def __set__(AdaDelta self, float val):
            self.o.rho = val
    property clip_abs:
        def __get__(AdaDelta self):
            return self.o.clip_abs
        def __set__(AdaDelta self, float val):
            self.o.clip_abs = val
    property clip_norm:
        def __get__(AdaDelta self):
            return self.o.clip_norm
        def __set__(AdaDelta self, float val):
            self.o.clip_norm = val
    property regc:
        def __get__(AdaDelta self):
            return self.o.regc
        def __set__(AdaDelta self, float val):
            self.o.regc = val
    property smooth_eps:
        def __get__(AdaDelta self):
            return self.o.smooth_eps
        def __set__(AdaDelta self, float val):
            self.o.smooth_eps = val

    def step(AdaDelta self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.step(params_c)

    def reset_caches(AdaDelta self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.reset_caches(params_c)

    def create_gradient_caches(AdaDelta self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.create_gradient_caches(params_c)

    def __cinit__(AdaDelta self,
                  params=None,
                  float rho=0.95,
                  float smooth_eps=1e-4,
                  float clip_abs=0.0,
                  float clip_norm=5.0,
                  float regc=0.0):
        cdef vector[CTensor] params_c
        if params is None:
            self.o = CAdaDelta(rho, smooth_eps, clip_norm, regc)
        else:
            params_c = ensure_tensor_list(params)
            self.o = CAdaDelta(params_c, rho, smooth_eps, clip_norm, regc)
        self.o.clip_abs = clip_abs

cdef class AdaGrad:
    @staticmethod
    cdef AdaGrad wrapc(CAdaGrad o):
        ret = AdaGrad()
        ret.o = o
        return ret

    property step_size:
        def __get__(AdaGrad self):
            return self.o.step_size
        def __set__(AdaGrad self, float val):
            self.o.step_size = val
    property clip_abs:
        def __get__(AdaGrad self):
            return self.o.clip_abs
        def __set__(AdaGrad self, float val):
            self.o.clip_abs = val
    property clip_norm:
        def __get__(AdaGrad self):
            return self.o.clip_norm
        def __set__(AdaGrad self, float val):
            self.o.clip_norm = val
    property regc:
        def __get__(AdaGrad self):
            return self.o.regc
        def __set__(AdaGrad self, float val):
            self.o.regc = val
    property smooth_eps:
        def __get__(AdaGrad self):
            return self.o.smooth_eps
        def __set__(AdaGrad self, float val):
            self.o.smooth_eps = val

    def step(AdaGrad self, params, step_size=None):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        if step_size is None:
            self.o.step(params_c)
        else:
            self.o.step_double(params_c, step_size)

    def reset_caches(AdaGrad self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.reset_caches(params_c)

    def create_gradient_caches(AdaGrad self, params):
        cdef vector[CTensor] params_c = ensure_tensor_list(params)
        self.o.create_gradient_caches(params_c)

    def __cinit__(AdaGrad self,
                  params=None,
                  float step_size=0.1,
                  float smooth_eps=1e-4,
                  float clip_abs=0.0,
                  float clip_norm=100.0,
                  float regc=0.0):
        cdef vector[CTensor] params_c
        if params is None:
            self.o = CAdaGrad(step_size, smooth_eps, clip_norm, regc)
        else:
            params_c = ensure_tensor_list(params)
            self.o = CAdaGrad(params_c, step_size, smooth_eps, clip_norm, regc)
        self.o.clip_abs = clip_abs

