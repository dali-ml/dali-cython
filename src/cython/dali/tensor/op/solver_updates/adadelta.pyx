def adadelta_update(Tensor t, Array gsum, Array xsum, double rho, double smooth_eps):
    """
    adagrad_update(t, gsum, xsum, rho, smooth_eps)

    Apply the AdaDelta update to `t.w` using the gradients in `t.dw`
    and the caches given as arguments.

    Parameters
    ----------
    t : Tensor containing gradients `t.dw` and parameters `t.w`
    gsum : Array, first cache
    xsum : Array, second cache
    rho : float, AdaDelta decay parameter
    smooth_eps : float, numerical stability parameters
                 (Note: this parameter matters in practice for
                  initial training stability, as division by small
                  numbers can lead to divergence and unstability)

    """
    c_adadelta_update(t.o, gsum.o, xsum.o, rho, smooth_eps)

__all__ = ["adadelta_update"]
