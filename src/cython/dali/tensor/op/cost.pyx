cpdef Tensor binary_cross_entropy(Tensor t, target):
    """
    binary_cross_entropy(t, target)

    Return the measure of the binary cross entropy between the
    input probabilities in `t` and a target distribution `target`.

    Parameters
    ----------
    t: input distribution, an N-dimensional tensor of 0-1 continuous values
    target: target distribution, an N-dimensional tensor of 0-1 continuous values

    Returns
    -------
    out : Tensor
        result of binary cross entropy measure

    See Also
    --------
    sigmoid_binary_cross_entropy : sigmoid followed by binary cross entropy
    """
    cdef Tensor target_c
    if isinstance(target, float):
        return Tensor.wrapc(cbinary_cross_entropy_double(t.o, target))
    else:
        target_c = ensure_tensor(target)
        return Tensor.wrapc(cbinary_cross_entropy(t.o, target_c.o))

cpdef Tensor sigmoid_binary_cross_entropy(Tensor t, target):
    """
    sigmoid_binary_cross_entropy(t, target)

    Return the result of applying sigmoid nonlinearity to `t`
    followed by measuring the binary cross entropy between the
    sigmoided values and a target distribution `target`.

    Parameters
    ----------
    t: input unnormalized log-likelihood N-dimensional Tensor
    target: target distribution, an N-dimensional tensor of 0-1 continuous values

    Returns
    -------
    out : Tensor
        result of sigmoid followed by binary cross entropy measure

    See Also
    --------
    binary_cross_entropy : binary cross entropy between two distributions
    """
    cdef Tensor target_c
    if isinstance(target, float):
        return Tensor.wrapc(csigmoid_binary_cross_entropy_double(t.o, target))
    else:
        target_c = ensure_tensor(target)
        return Tensor.wrapc(csigmoid_binary_cross_entropy(t.o, target_c.o))

cpdef Tensor margin_loss(Tensor t, target, double margin, int axis):
    """
    margin_loss(t, target, margin, axis)

    Computes a form of hinge loss between the inputs and the target
    values.

    """
    cdef Tensor target_c
    if isinstance(target, int):
        return Tensor.wrapc(cmargin_loss_int(t.o, target, margin, axis))
    else:
        target_c = ensure_tensor(target)
        return Tensor.wrapc(cmargin_loss(t.o, target_c.o, margin, axis))

cpdef Tensor softmax_cross_entropy(Tensor logits, Tensor target, double temperature=1.0, int axis=-1):
    """
    softmax_cross_entropy(t, target, temperature=1.0, axis=-1)

    Return a tensor that is the result of exponential normalization followed
    by Kullback-Leibler divergence with a target distribution.

    Parameters
    ----------
    logits: input unnormalized log-likelihood N-dimensional Tensor
    target: target distribution, an N-dimensional tensor of distributions
            or (N-1)-dimensional integer Tensor representing one-hot
            distributions along the `axis` dimension.
    temperature : a hyperparameter that controls the rolloff of the exponential.
                  when this value is high, the difference between the maximum
                  and the minimum value is smaller, while a low temperature
                  will make the differences greater.
    Returns
    -------
    out : Tensor
        result of Softmax operation followed by Kullback-Leibler divergence

    See Also
    --------
    softmax : exponential normalization
    cross_entropy : Kullback-Leibler divergence between two distributions
    """
    return Tensor.wrapc(csoftmax_cross_entropy(logits.o, target.o, temperature, axis))

cpdef Tensor softmax(Tensor t, int axis=-1, double temperature=1.0):
    """
    softmax(t, axis=-1, temperature=1.0)

    Return a tensor that is the result of exponential normalization,
    more commonly known as Softmax:

    ``out[...,i,...] = exp(t[...,i,...] / temperature) / exp(t / temperature ).sum(axis)``

    Parameters
    ----------
    axis : along what dimension should normalization occur. Defaults to -1.
    temperature : a hyperparameter that controls the rolloff of the exponential.
                  when this value is high, the difference between the maximum
                  and the minimum value is smaller, while a low temperature
                  will make the differences greater.
    Returns
    -------
    out : Tensor
        result of Softmax operation

    See Also
    --------
    cross_entropy : Kullback-Leibler divergence between two distributions
    softmax_cross_entropy : exponential normalization followed by
                            Kullback-Leibler divergence with a target distribution
    """
    return Tensor.wrapc(csoftmax(t.o, axis, temperature))

cpdef Tensor cross_entropy(Tensor probs, Tensor target, int axis=-1):
    """
    cross_entropy(probs, target, axis=-1)

    Return the element-wise Kullback-Leibler divergence of probs
    under the target distribution. This function can also be used
    to compute sparse_cross_entropy by passing an integer set of
    targets that define one-hot distributions in the axis
    given by `axis`.

    Parameters
    ----------
    axis : optional; when performing cross_entropy using integer targets, along
           what dimension should each target represent a one-hot distribution.
           Defaults to -1.

    Returns
    -------
    out : Tensor
        result of Kullback-Leibler divergence.

    See Also
    --------
    softmax : exponential normalization
    softmax_cross_entropy : exponential normalization followed by
                            Kullback-Leibler divergence with a target distribution
    """
    return Tensor.wrapc(ccross_entropy(probs.o, target.o, axis))
