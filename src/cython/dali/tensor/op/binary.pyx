
cpdef Tensor add_n(tensors):
    """
    add_n(tensors)

    Return the element-wise sum of all input tensors.

    Arguments
    ---------
    tensors : a list or tuple of inputs of type Tensor

    Returns
    -------
    out : Tensor
        result of the element-wise sum of all input tensors
    """
    cdef vector[CTensor] tensors_c
    cdef Tensor tensor_c

    got_list_of_tensors = (
        isinstance(tensors, (tuple, list)) and
        all([type(t) == Tensor for t in tensors])
    )

    if not got_list_of_tensors:
        raise ValueError("tensors should be a list or a tuple of tensors")

    for tensor in tensors:
        tensor_c = tensor
        tensors_c.emplace_back(tensor_c.o)

    return Tensor.wrapc(cadd(tensors_c))

cpdef Tensor add(Tensor left, Tensor right):
    """
    add(left, right)

    Return the element-wise addition of left and right tensors.

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

