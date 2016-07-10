
cpdef Tensor add_n(tensors):
    """
    add_n(tensors)

    Return the element-wise sum of all input tensors.

    ``out[i] = tensors[0][i] + tensors[1][i] + ... + tensors[n][i]``

    Arguments
    ---------
    tensors : a list or tuple of inputs of type Tensor

    Returns
    -------
    out : Tensor
        result of the element-wise sum of all input tensors
    """
    return Tensor.wrapc(cadd(ensure_tensor_list(tensors)))

cpdef Tensor add(Tensor left, Tensor right):
    """
    add(left, right)

    Return the element-wise addition of left and right tensors.

    ``out[i] = left[i] + right[i]``

    Arguments
    ---------
    left : an N-dimensional tensor
    right : an N-dimensional tensor

    Returns
    -------
    out : Tensor
        result of the element-wise sum of left and right tensors
    """
    return Tensor.wrapc(cadd(left.o, right.o))

cpdef Tensor sub(Tensor left, Tensor right):
    """
    sub(left, right)

    Return the element-wise substraction of right from left

    ``out[i] = left[i] - right[i]``

    Arguments
    ---------
    left : an N-dimensional tensor
    right : an N-dimensional tensor

    Returns
    -------
    out : Tensor
        result of the element-wise substraction of right from left
    """
    return Tensor.wrapc(csub(left.o, right.o))

cpdef Tensor eltmul(Tensor left, Tensor right):
    """
    eltmul(left, right)

    Return the element-wise multiplication of left and right

    ``out[i] = left[i] * right[i]``

    Arguments
    ---------
    left : an N-dimensional tensor
    right : an N-dimensional tensor

    Returns
    -------
    out : Tensor
        result of the element-wise substraction of right from left
    """
    return Tensor.wrapc(celtmul(left.o, right.o))

cpdef Tensor eltdiv(Tensor left, Tensor right):
    """
    eltdiv(left, right)

    Return the element-wise division of left by right

    ``out[i] = left[i] / right[i]``

    Arguments
    ---------
    left : an N-dimensional tensor
    right : an N-dimensional tensor

    Returns
    -------
    out : Tensor
        result of the element-wise division of left by right
    """
    return Tensor.wrapc(celtdiv(left.o, right.o))

cpdef Tensor pow(Tensor x, Tensor exponent):
    """
    pow(x, exponent)

    Return the element-wise raise to the power of x by exponent

    ``out[i] = x[i] ** exponent[i]``

    Arguments
    ---------
    x : an N-dimensional tensor
    exponent : an N-dimensional tensor

    Returns
    -------
    out : Tensor
        result of the element-wise raise-to-the-power of x by exponent
    """
    return Tensor.wrapc(cpow(x.o, exponent.o))

cpdef Tensor circular_convolution(Tensor content, Tensor shift):
    """
    circular_convolution(content, shift)

    Return the element-wise circular-convolution of content by shift.

    In 1D this can be written as:
    ``
    for i in content.shape[0]:
      for j in content.shape[0]:
        offset = i + j % content.shape[0]
        out[i] += content[offset] * shift[j]
    ``

    Arguments
    ---------
    content : an N-dimensional tensor
    shift : an N-dimensional tensor

    Returns
    -------
    out : Tensor
        result of the circular-convolution of content by shift
    """
    return Tensor.wrapc(ccircular_convolution(content.o, shift.o))

cpdef Tensor prelu(Tensor x, Tensor weights):
    """
    prelu(x, weights)

    Return the parametric-relu of x using weights.
    This is a function that returns identity for positive x,
    and the product of weights by x when negative.

    In 1D this can be written as:
    ``out[i] = x[i] > 0 ? x[i] : weight[i] * x[i]``

    Arguments
    ---------
    content : an N-dimensional tensor
    shift : an N-dimensional tensor

    Returns
    -------
    out : Tensor
        result of the prelu activation of x using weights.
    """
    return Tensor.wrapc(cprelu(x.o, weights.o))
