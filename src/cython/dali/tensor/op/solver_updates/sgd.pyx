def sgd_update(Tensor t, double step_size):
    """
    sgd_update(t, step_size)

    Modify the Array `t.w` by substracting the Array `t.dw * step_size`

    Parameters
    ----------
    t : Tensor that holds the weights and gradients
    step_size : float, by how much to subtract the gradient from `t.dw`
    """
    c_sgd_update(t.o, step_size)

__all__ = ["sgd_update"]
