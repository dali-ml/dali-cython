def rmsprop_update(
        Tensor t,
        Array cache,
        double decay_rate,
        double step_size,
        double smooth_eps):
    """
    rmsprop_update(t, cache, decay_rate, step_size, smooth_eps)

    Apply the RMSProp update to `t.w` using the gradients in `t.dw`
    and the cache given as argument.
    """
    c_rmsprop_update(t.o, cache.o, decay_rate, step_size, smooth_eps)

def rmsprop_momentum_update(
        Tensor t,
        Array n_cache,
        Array g_cache,
        Array momentum_cache,
        double decay_rate,
        double momentum,
        double step_size,
        double smooth_eps):
    """
    rmsprop_momentum_update(t, n_cache, g_cache, momentum_cache, decay_rate, momentum, step_size, smooth_eps)

    Apply the RMSProp with momentum update to `t.w` using the gradients in `t.dw`
    and the caches given as arguments.
    """
    c_rmsprop_momentum_update(t.o, n_cache.o, g_cache.o, momentum_cache.o, decay_rate, momentum, step_size, smooth_eps)

__all__ = ["rmsprop_update", "rmsprop_momentum_update"]
