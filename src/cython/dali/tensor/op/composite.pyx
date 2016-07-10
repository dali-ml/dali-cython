cpdef Tensor dot_with_bias(Tensor inputs, Tensor weights, Tensor bias):
    """
    dot_with_bias(inputs, weights, bias)

    Return the result of ``(inputs • weights) + bias``

    Arguments:
    ----------
    inputs : a 2-dimensional Tensor
    weights : a 2-dimensional Tensor
    bias : a 2-dimensional Tensor, with first dimension broadcasted or
           equal to the leading dimension of inputs.

    Returns:
    --------
    out : Tensor
        the result of the dot product of inputs with weights added to bias
    """
    return Tensor.wrapc(cdot_with_bias(inputs.o, weights.o, bias.o))

cpdef Tensor multiple_dot_with_bias(inputs, weights, Tensor bias):
    """
    multiple_dot_with_bias(inputs, weights, bias)

    Return the result of:
    ``(inputs[0] • weights[0] + ... + inputs[n] • weights[n]) + bias``

    Arguments:
    ----------
    inputs : a list of 2-dimensional Tensor
    weights : a list of 2-dimensional Tensor
    bias : a 2-dimensional Tensor, with first dimension broadcasted or
           equal to the leading dimension of each Tensor in inputs.

    Returns:
    --------
    out : Tensor
        the result of the sum of bias with all the dot products of each input with weight
    """
    return Tensor.wrapc(
        cmultiple_dot_with_bias(
            ensure_tensor_list(inputs),
            ensure_tensor_list(weights),
            bias.o
        )
    )

cpdef Tensor quadratic_form(Tensor left, Tensor middle, Tensor right):
    """
    quadratic_form(left, middle, right)

    Return the result of quadratic-form: (left.T • middle) • right

    Arguments:
    ----------
    left : an N-dimensional Tensor
    middle : an N-dimensional Tensor
    right : an N-dimensional Tensor

    Returns:
    --------
    out : Tensor
        the result of the quadratic-form of left, middle, right.
    """
    return Tensor.wrapc(cquadratic_form(left.o, middle.o, right.o))
