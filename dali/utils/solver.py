import copy

from dali.core import Mat, MatOps


class Solver(object):
    known_solvers = [
        'sgd',
        'adagrad',
        'rmsprop',
        'adadelta',
        'adam',
    ]
    known_params = [
        'learning_rate',
        'clipval',
        'regc',
        'smooth_eps',
        'rho',
        'b1',
        'b2',
        'decay_rate',
    ]

    def __init__(self, solver_type, **kwargs):
        self.solver_type = solver_type
        if solver_type not in Solver.known_solvers:
            raise AttributeError("Unknown solver " + str(solver_type))
        self.kwargs = kwargs
        for key in kwargs:
            if key not in Solver.known_params:
                raise AttributeError("Unknown keyword argument " + key)

    def get_arg(self, override, name, default_val):
        if name in override:
            return override[name]
        if name in self.kwargs:
            return self.kwargs[name]
        return default_val

    def step(self, param, **kwargs_override):
        if type(param) == list:
            for p in param:
                self.step(p, **kwargs_override)
        elif type(param) == Mat:
            for key in kwargs_override:
                if not key in Solver.known_params:
                    raise AttributeError("Unknown keyword argument " + key)

            clip_val = self.get_arg(kwargs_override, 'clipval', 5.0)
            regc     = self.get_arg(kwargs_override, 'regc',     0.0)

            MatOps.clip_and_regularize(param, clip_val, regc)

            learning_rate = self.get_arg(kwargs_override, "learning_rate", 0.01)

            lr_multiplier = param.extra_state.get('lr_multiplier', 1.0)
            learning_rate *= lr_multiplier

            if self.solver_type == 'sgd':
                MatOps.sgd_update(param, learning_rate)
            elif self.solver_type == 'adagrad':
                smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-6)
                cache = self.get_cache(param, 'adagrad_cache')
                MatOps.adagrad_update(param, cache, learning_rate, smooth_eps)
            elif self.solver_type == 'rmsprop':
                smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-6)
                decay_rate = self.get_arg(kwargs_override, "decay_rate", 0.95)
                cache = self.get_cache(param, 'rmsprop_cache')
                MatOps.rmsprop_update(param, cache, decay_rate, learning_rate, smooth_eps)
            elif self.solver_type == 'adadelta':
                smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-4)
                rho        = self.get_arg(kwargs_override, "rho",        0.95)
                gsum = self.get_cache(param, 'adadelta_gsum')
                xsum = self.get_cache(param, 'adadelta_xsum')
                MatOps.adadelta_update(param, gsum, xsum, rho, smooth_eps)
            elif self.solver_type == 'adam':
                smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-4)
                b1         = self.get_arg(kwargs_override, "b1",        0.5)
                b2         = self.get_arg(kwargs_override, "b2",        1e-6)
                m  = self.get_cache(param, 'adam_m')
                v  = self.get_cache(param, 'adam_v')
                epoch = param.extra_state.get('adam_epoch', 1)

                MatOps.adam_update(param, m, v, b1, b2, smooth_eps, learning_rate, epoch)

                param.extra_state['adam_epoch'] = epoch + 1
            else:
                assert False
            param.clear_grad()
        else:
            raise AttributeError("step accepts list or tensor")

    def set_lr_multiplier(self, param, lr_multiplier):
        param.extra_state["lr_multiplier"] = lr_multiplier

    def get_cache(self, param, cache_name):
        if cache_name not in param.extra_state:
            param.extra_state[cache_name] = Mat.zeros(param.shape, dtype=param.dtype)
        return param.extra_state[cache_name]

    def reset_caches(self, param):
        if type(param) == list:
            for param in list:
                self.reset_caches(param)
        elif type(param) == Mat:
            if self.solver_type == 'adagrad':
                self.get_cache(param, 'adagrad_cache').clear()
            elif self.solver_type == 'rmsprop':
                self.get_cache(param, 'rmsprop_cache').clear()
            elif self.solver_type == 'adadelta':
                self.get_cache(param, 'adadelta_gsum').clear()
                self.get_cache(param, 'adadelta_xsum').clear()
            elif self.solver_type == 'adam':
                self.get_cache(param, 'adam_m').clear()
                self.get_cache(param, 'adam_v').clear()
                if 'adam_epoch' in param:
                    del param.extra_state["adam_epoch"]
            else:
                assert False
