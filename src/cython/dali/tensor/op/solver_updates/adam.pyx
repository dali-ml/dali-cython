def adam_update(Tensor t,
                Array m,
                Array v,
                double b1,
                double b2,
                double smooth_eps,
                double step_size,
                int epoch):
    """
    adam_update(t, m, v, b1, b2, smooth_eps, step_size, epoch)

    Apply the Adam update to `t.w` using the gradients in `t.dw`
    and the caches given as arguments.
    """
    c_adam_update(t.o, m.o, v.o, b1, b2, smooth_eps, step_size, epoch)

__all__ = ["adam_update"]
