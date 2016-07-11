cpdef AssignableArray add_n(arrays):
    """
    add_n(arrays)

    Return the element-wise sum of all input arrays.

    ``out[i] = arrays[0][i] + arrays[1][i] + ... + arrays[n][i]``

    Parameters
    ----------
    arrays : a list or tuple of inputs of type Array

    Returns
    -------
    out : Array
        result of the element-wise sum of all input arrays
    """
    return AssignableArray.wrapc(c_add(ensure_array_list(arrays)))

cpdef AssignableArray add(Array left, Array right):
    """
    add(left, right)

    Return the element-wise addition of left and right arrays.

    ``out[i] = left[i] + right[i]``

    Parameters
    ----------
    left : an N-dimensional array
    right : an N-dimensional array

    Returns
    -------
    out : Array
        result of the element-wise sum of left and right arrays
    """
    return AssignableArray.wrapc(c_add(left.o, right.o))

cpdef AssignableArray sub(Array left, Array right):
    """
    sub(left, right)

    Return the element-wise substraction of right from left

    ``out[i] = left[i] - right[i]``

    Parameters
    ----------
    left : an N-dimensional array
    right : an N-dimensional array

    Returns
    -------
    out : Array
        result of the element-wise substraction of right from left
    """
    return AssignableArray.wrapc(c_sub(left.o, right.o))

cpdef AssignableArray eltmul(Array left, Array right):
    """
    eltmul(left, right)

    Return the element-wise multiplication of left and right

    ``out[i] = left[i] * right[i]``

    Parameters
    ----------
    left : an N-dimensional array
    right : an N-dimensional array

    Returns
    -------
    out : Array
        result of the element-wise substraction of right from left
    """
    return AssignableArray.wrapc(c_eltmul(left.o, right.o))

cpdef AssignableArray eltdiv(Array left, Array right):
    """
    eltdiv(left, right)

    Return the element-wise division of left by right

    ``out[i] = left[i] / right[i]``

    Parameters
    ----------
    left : an N-dimensional array
    right : an N-dimensional array

    Returns
    -------
    out : Array
        result of the element-wise division of left by right
    """
    return AssignableArray.wrapc(c_eltdiv(left.o, right.o))

cpdef AssignableArray pow(Array x, Array exponent):
    """
    pow(x, exponent)

    Return the element-wise raise to the power of x by exponent

    ``out[i] = x[i] ** exponent[i]``

    Parameters
    ----------
    x : an N-dimensional array
    exponent : an N-dimensional array

    Returns
    -------
    out : Array
        result of the element-wise raise-to-the-power of x by exponent
    """
    return AssignableArray.wrapc(c_pow(x.o, exponent.o))

cpdef AssignableArray circular_convolution(Array content, Array shift):
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

    Parameters
    ----------
    content : an N-dimensional array
    shift : an N-dimensional array

    Returns
    -------
    out : Array
        result of the circular-convolution of content by shift
    """
    return AssignableArray.wrapc(c_circular_convolution(content.o, shift.o))

cpdef AssignableArray prelu(Array x, Array weights):
    """
    prelu(x, weights)

    Return the parametric-relu of x using weights.
    This is a function that returns identity for positive x,
    and the product of weights by x when negative.

    In 1D this can be written as:
    ``out[i] = x[i] > 0 ? x[i] : weight[i] * x[i]``

    Parameters
    ----------
    content : an N-dimensional array
    shift : an N-dimensional array

    Returns
    -------
    out : Array
        result of the prelu activation of x using weights.
    """
    return AssignableArray.wrapc(c_prelu(x.o, weights.o))


cpdef AssignableArray equals(Array left, Array right):
    """
    equals(left, right)

    Returns a binary array representing elementwise equality
    of left and right, i.e. if elements at position i are
    the same for left and right, the result will containt
    one at that position and zero otherwise:

        out[i] = int(left[i] == right[i])

    Parameters
    ----------
    left : an N-dimensional array
    right : an N-dimensional array

    Returns
    -------
    out : Array
        result of the equals operation
    """
    return AssignableArray.wrapc(c_equals(left.o, right.o))
