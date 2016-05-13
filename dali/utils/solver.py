import copy
from dali.core import Mat, MatOps
from dali.utils.throttled import Throttled

class SolverBase(object):
    t = Throttled(1)
    known_solvers = [
        'sgd',
        'adagrad',
        'rmsprop',
        'rmsprop_momentum',
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
        'debug',
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

    def step(self, params, param_caches=None, **kwargs_override):
        if type(params) != list:
            params = [params]
            if param_caches is not None:
                param_caches = [param_caches]

        if param_caches is None:
            param_caches = [None for _ in range(params)]
        assert len(params) == len(param_caches)

        for key in kwargs_override:
            if not key in SolverBase.known_params:
                raise AttributeError("Unknown keyword argument " + key)

        debug = self.get_arg(kwargs_override, 'debug', ['nans'])
        clip_val = self.get_arg(kwargs_override, 'clipval', 5.0)
        regc     = self.get_arg(kwargs_override, 'regc',     0.0)

        ########## GRADIENT NORMALIZATION ###########

        gradient_normalization = self.get_arg(kwargs_override, 'gradient_normalization', 'norm')
        if gradient_normalization == 'norm':
            for param in params:
                MatOps.clip_and_regularize(param, 0.0, clip_val, regc)
        elif gradient_normalization == 'clipping':
            for param in params:
                MatOps.clip_and_regularize(param, clip_val, 0.0, regc)
        elif gradient_normalization == 'discard':
            params_exceeding = []
            for param in params:
                if MatOps.grad_norm(param).w[0,0] > clip_val:
                    params_exceeding.append(param.name if param.name != '' else '(unnamed parameter)')
            if len(params_exceeding) > 0:
                if 'discards' in debug:
                    print('Discarding gradient update due to exceeded norm for the following parameters: %s' % (params_exceeding,))
                for param in params:
                    param.clear_grad()
                return
        elif gradient_normalization == 'none':
            if regc > 0.0:
                MatOps.regularize(param, regc)
        else:
            raise AttributeError("Unknown gradient_normalization mode : " + gradient_normalization)


        ########## SOLVING ###########

        for param, param_cache in zip(params, param_caches):
            if param_cache is None:
                param_cache = self.param_to_cache(param)

            learning_rate = self.get_arg(kwargs_override, "learning_rate", 0.01)

            lr_multiplier = param.extra_state.get('lr_multiplier', 1.0)
            learning_rate *= lr_multiplier
            if MatOps.is_grad_nan(param):
                if SolverBase.t.should_i_run() and 'nans' in debug:
                    name_str = ' (unnamed parameter)'
                    if param.name is not None:
                        name_str = ' (name: %s)' % (param.name,)
                    print("Warning ignoring grad update due to NaNs%s." % (name_str,))
            else:
                if self.solver_type == 'sgd':
                    MatOps.sgd_update(param, learning_rate)
                elif self.solver_type == 'adagrad':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-6)
                    cache = self.get_cache(param, param_cache, 'adagrad_cache')
                    MatOps.adagrad_update(param, cache, learning_rate, smooth_eps)
                elif self.solver_type == 'rmsprop':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-6)
                    decay_rate = self.get_arg(kwargs_override, "decay_rate", 0.95)
                    cache = self.get_cache(param, param_cache, 'rmsprop_cache')
                    MatOps.rmsprop_update(param, cache, decay_rate, learning_rate, smooth_eps)
                elif self.solver_type == 'rmsprop_momentum':
                    decay_rate = self.get_arg(kwargs_override,    "decay_rate", 0.95)
                    momentum = self.get_arg(kwargs_override,      "momentum",   0.9)
                    smooth_eps = self.get_arg(kwargs_override,    "smooth_eps", 1e-4)
                    n_cache = self.get_cache(param, param_cache, 'rmsprop_momentum_n_cache')
                    g_cache = self.get_cache(param, param_cache, 'rmsprop_momentum_g_cache')
                    momentum_cache = self.get_cache(param, param_cache, 'rmsprop_momentum_momentum_cache')
                    MatOps.rmsprop_momentum_update(param, n_cache, g_cache, momentum_cache, decay_rate, momentum, learning_rate, smooth_eps)
                elif self.solver_type == 'adadelta':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-4)
                    rho        = self.get_arg(kwargs_override, "rho",        0.95)
                    gsum = self.get_cache(param, param_cache, 'adadelta_gsum')
                    xsum = self.get_cache(param, param_cache, 'adadelta_xsum')
                    MatOps.adadelta_update(param, gsum, xsum, rho, smooth_eps)
                elif self.solver_type == 'adam':
                    smooth_eps = self.get_arg(kwargs_override, "smooth_eps", 1e-4)
                    b1         = self.get_arg(kwargs_override, "b1",        0.5)
                    b2         = self.get_arg(kwargs_override, "b2",        1e-6)
                    m  = self.get_cache(param, param_cache, 'adam_m')
                    v  = self.get_cache(param, param_cache, 'adam_v')
                    epoch = param.extra_state.get('adam_epoch', 1)

                    MatOps.adam_update(param, m, v, b1, b2, smooth_eps, learning_rate, epoch)

                    param_cache['adam_epoch'] = epoch + 1
                else:
                    assert False
            param.clear_grad()

    def set_lr_multiplier(self, param, lr_multiplier):
        param.extra_state["lr_multiplier"] = lr_multiplier

    def get_cache(self, param, cache_state, cache_name):
        if cache_name not in cache_state:
            cache_state[cache_name] = Mat.zeros(param.shape, dtype=param.dtype)
        ret = cache_state[cache_name]
        assert ret.shape == param.shape, \
                "Wrong parameter passed to solver (cache shape does not match parameter's shape)"
        return ret

    def reset_caches(self, param, param_caches=None):
        if type(param) == list:
            if param_caches is not None:
                assert len(param) == len(param_caches)
                for p,c  in zip(param, param_caches):
                    self.reset_caches(p, c)
            else:
                for p in param:
                    self.reset_caches(p)
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
            self._parameters = parameters
        else:
            self._parameters = [parameters]
        self.caches =  [{} for _ in range(len(self._parameters))]

        self.lr_multipliers = [None for _ in range(len(self._parameters))]

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        assert self._parameters is None or \
                len(self._parameters) == len(val), \
                "Number of parameters must remain unchanged"
        self._parameters = val
        for lr_multiplier, param in zip(self.lr_multipliers, self._parameters):
            if lr_multiplier is not None:
                self.base.set_lr_multiplier(param, lr_multiplier)

    def set_lr_multiplier(self, where, val):
        indices = []
        if type(where) == str:
            for i, param in enumerate(self.parameters):
                if param.name == where:
                    indices.append(i)
            assert len(indices) > 0, \
                    "Could not find parameters %s" % (where,)
        elif type(where) == int:
            indices.append(where)
        else:
            raise ValueError("where must be int or str")
        for i in indices:
            self.lr_multipliers[i] = val
            self.base.set_lr_multiplier(self.parameters[i], val)

    def step(self):
        assert self.parameters is not None, \
                "Remeber to use set parameters after unpickling."
        self.base.step(self.parameters, self.caches)

    def reset_caches(self):
        assert self.parameters is not None, \
                "Remeber to use set parameters after unpickling."
        self.base.reset_caches(self.parameters, self.caches)

    @property
    def solver_type(self):
        return self.base.solver_type

    def __setstate__(self, state):
        self.base          = state['solver']
        self.caches       = state['caches']
        self.lr_multipliers = state['lr_multipliers']
        self._parameters = None

    def __getstate__(self):
        return {
            'solver'      : self.base,
            'caches'      : self.caches,
            'lr_multipliers': self.lr_multipliers
        }

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
