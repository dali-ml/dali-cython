def clip_and_regularize(Tensor t,
                        double clip_abs,
                        double clip_norm,
                        double regc):
    """
    clip_and_regularize(t, clip_abs, clip_norm, regc)

    Ensure that the norm of the gradient in `t.dw` has
    no value larger than `clip_abs` and a euclidean norm
    less than `clip_norm`. Additionally adds the gradient
    for the L2 Norm of `t.w` to the gradient.

    Parameters
    ----------
    t : Tensor with the gradient t.dw to regularize
    clip_abs : float, maximum absolute value of any gradient
               element
    clip_norm : float, maximum euclidean norm of gradient
    regc : float, weight of the L2 regularization term

    """
    c_clip_and_regularize(t.o, clip_abs, clip_norm, regc)

def regularize(Tensor t, double regc):
    """
    regularize(t, regc)

    Add the L2 norm gradient to `t.dw`
    (e.g. add `t.w * regc` to `t.dw`).
    This function does nothing for regc <= 0

    Parameters
    ----------
    t : Tensor with the gradient t.dw to regularize
    regc : float, weight of the L2 regularization term
    """
    c_regularize(t.o, regc)


def normalize_gradient(Tensor t, double norm_threshold):
    """
    normalize_gradient(t, norm_threshold)

    Ensure that the gradient `t.dw` has a euclidean
    norm inferior or equal to `norm_threshold`

    Parameters
    ----------
    t : Tensor with the gradient t.dw to normalize
    norm_threshold : float, largest acceptable norm
    """
    c_normalize_gradient(t.o, norm_threshold)

__all__ = ["clip_and_regularize", "regularize", "normalize_gradient"]
