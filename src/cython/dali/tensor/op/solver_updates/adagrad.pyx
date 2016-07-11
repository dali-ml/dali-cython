def adagrad_update(Tensor t, Array cache, double step_size, double smooth_eps):
    """
    adagrad_update(t, cache, step_size, smooth_eps)

    Apply the Adagrad update to `t.w` using the gradients in `t.dw`
    and the caches given as arguments.

    Parameters
    ----------
    t : Tensor containing gradients `t.dw` and parameters `t.w`
    cache : Array, cache of sum of gradient squares (variance estimate)
    step_size : float, size of a step (with cache providing a more granular
    		    per parameter step_size)
    smooth_eps : float, numerical stability parameters
                 (Note: this parameter matters in practice for
                  initial training stability, as division by small
                  numbers can lead to divergence and unstability)
    """
    c_adagrad_update(t.o, cache.o, step_size, smooth_eps)

__all__ = ["adagrad_update"]
