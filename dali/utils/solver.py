import copy
from dali.core import Mat, MatOps
from dali.utils.throttled import Throttled

class SolverBase(object):
    t = Throttled(1)
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
        'gradient_normalization',
    ]

    def __init__(self, solver_type, **kwargs):
        self.solver_type = solver_type
        if solver_type not in SolverBase.known_solvers:
            raise AttributeError("Unknown solver " + str(solver_type))
        self.kwargs = kwargs
        for key in kwargs:
            if key not in SolverBase.known_params:
                raise AttributeError("Unknown keyword argument " + key)

    def get_arg(self, override, name, default_val):
        if name in override:
            return override[name]
        if name in self.kwargs:
            return self.kwargs[name]
        return default_val

    def param_to_cache(self, param):
        if 'solver_cache' not in param.extra_state:
            param.extra_state['solver_cache'] = {}
        return param.extra_state['solver_cache']

    def step(self, param, param_caches=None, **kwargs_override):
        if type(param) == list:
            if param_caches is not None:
                assert len(param) == len(param_caches)
                for p,c  in zip(param, param_caches):
                    self.step(p, c, **kwargs_override)
            else:
                for p in param:
                    self.step(p, **kwargs_override)
        elif type(param) == Mat:
            if param_caches is None:
                param_caches = self.param_to_cache(param)
            for key in kwargs_override:
                if not key in SolverBase.known_params:
                    raise AttributeError("Unknown keyword argument " + key)

            clip_val = self.get_arg(kwargs_override, 'clipval', 5.0)
            regc     = self.get_arg(kwargs_override, 'regc',     0.0)
            gradient_normalization = self.get_arg(kwargs_override, 'gradient_normalization', 'norm')
            if gradient_normalization == 'norm':
                if regc > 0.0:
                    MatOps.regularize(param, regc)
                MatOps.normalize(param, clip_val)
            elif gradient_normalization == 'clipping':
                MatOps.clip_and_regularize(param, clip_val, regc)
            elif gradient_normalization == 'none':
                if regc > 0.0:
                    MatOps.regularize(param, regc)
            else:
                raise AttributeError("Unknown gradient_normalization mode : " + gradient_normalization)


            learning_rate = self.get_arg(kwargs_override, "learning_rate", 0.01)

            lr_multiplier = param.extra_state.get('lr_multiplier', 1.0)
            learning_rate *= lr_multiplier
            if MatOps.is_grad_nan(param):
                if SolverBase.t.should_i_run():
                    print("Warning ignoring grad update due to NaNs.")
            else:
                if self.solver_type == 'sgd':
                    MatOps.sgd_update(param, learning_rate)
                elif self.solver_type == 'adagrad':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-6)
                    cache = self.get_cache(param, param_caches, 'adagrad_cache')
                    MatOps.adagrad_update(param, cache, learning_rate, smooth_eps)
                elif self.solver_type == 'rmsprop':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-6)
                    decay_rate = self.get_arg(kwargs_override, "decay_rate", 0.95)
                    cache = self.get_cache(param, param_caches, 'rmsprop_cache')
                    MatOps.rmsprop_update(param, cache, decay_rate, learning_rate, smooth_eps)
                elif self.solver_type == 'adadelta':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-4)
                    rho        = self.get_arg(kwargs_override, "rho",        0.95)
                    gsum = self.get_cache(param, param_caches, 'adadelta_gsum')
                    xsum = self.get_cache(param, param_caches, 'adadelta_xsum')
                    MatOps.adadelta_update(param, gsum, xsum, rho, smooth_eps)
                elif self.solver_type == 'adam':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-4)
                    b1         = self.get_arg(kwargs_override, "b1",        0.5)
                    b2         = self.get_arg(kwargs_override, "b2",        1e-6)
                    m  = self.get_cache(param, param_caches, 'adam_m')
                    v  = self.get_cache(param, param_caches, 'adam_v')
                    epoch = param.extra_state.get('adam_epoch', 1)

                    MatOps.adam_update(param, m, v, b1, b2, smooth_eps, learning_rate, epoch)

                    param_caches['adam_epoch'] = epoch + 1
                else:
                    assert False
            param.clear_grad()
        else:
            raise AttributeError("step accepts list or tensor")

    def set_lr_multiplier(self, param, lr_multiplier):
        param.extra_state["lr_multiplier"] = lr_multiplier

    def get_cache(self, param, cache_state, cache_name):
        if cache_name not in cache_state:
            cache_state[cache_name] = Mat.zeros(param.shape, dtype=param.dtype)
        ret = cache_state[cache_name]
        assert ret.shape == param.shape, \
                "Wrong parameter passed to solver (cache shape not matching the parameter"
        return ret

    def reset_caches(self, param, param_caches=None):
        if type(param) == list:
            if param_caches is not None:
                assert len(param) == len(param_caches)
                for p,c  in zip(param, param_caches):
                    self.reset_caches(p, c, **kwargs_override)
            else:
                for p in param:
                    self.reset_caches(p, **kwargs_override)
        elif type(param) == Mat:
            # get caches
            if param_caches is None:
                param_caches = self.param_to_cache(param)
            # reset
            if self.solver_type == 'adagrad':
                self.get_cache(param, param_caches, 'adagrad_cache').clear()
            elif self.solver_type == 'rmsprop':
                self.get_cache(param, param_caches, 'rmsprop_cache').clear()
            elif self.solver_type == 'adadelta':
                self.get_cache(param, param_caches, 'adadelta_gsum').clear()
                self.get_cache(param, param_caches, 'adadelta_xsum').clear()
            elif self.solver_type == 'adam':
                self.get_cache(param, param_caches, 'adam_m').clear()
                self.get_cache(param, param_caches, 'adam_v').clear()
                if 'adam_epoch' in param:
                    del param_caches["adam_epoch"]
            else:
                assert False

class Solver(object):
    def __init__(self, parameters, *args, **kwargs):
        """Solver

        It is pickleable.
        """
        self.base = SolverBase(*args, **kwargs)
        if type(parameters) == list:
            self.parameters = parameters
        else:
            self.parameters = [parameters]
        self.caches =  [{} for _ in range(len(self.parameters))]

    def step(self):
        self.base.step(self.parameters, self.caches)

    def reset_caches(self):
        self.base.reset_caches(self.parameters, self.caches)

class CombinedSolver(object):
    def __init__(self, solvers):
        self.solvers = solvers

    def step(self):
        for solver in self.solvers:
            solver.step()

    def reset_caches(self):
        for solver in self.solvers:
            solver.reset_caches()

__all__ = [
    "CombinedSolver",
    "SolverBase",
    "Solver",
]
