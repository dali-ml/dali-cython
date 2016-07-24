cpdef Tensor concatenate(tensors, int axis):
    """
    concatenate(tensors)

    Arguments
    ---------
    tensors : a list or tuple of inputs of type Tensor
    axis : integer, axis along which to concatenate the tensors.

    Returns
    -------
    out : Tensor
        result of the reshape operation.
    """
    return Tensor.wrapc(c_concatenate(ensure_tensor_list(tensors), axis))

cpdef Tensor hstack(tensors):
    """
    hstack(tensors)

    Stack arrays in sequence horizontally (column wise).

    Take a sequence of tensors and stack them horizontally to make
    a single Tensor.

    Parameters
    ----------
    tup : sequence of tensors
        All arrays must have the same shape along all but the second axis.

    Returns
    -------
    stacked : Tensor
        The Tensor formed by stacking the given tensors.

    See Also
    --------
    stack : Join a sequence of arrays along a new axis.
    vstack : Stack arrays in sequence vertically (row wise).
    concatenate : Join a sequence of arrays along an existing axis.

    Notes
    -----
    Equivalent to ``dali.concatenate(tup, axis=1)``

    Examples
    --------
    >>> a = dali.Tensor((1,2,3))
    >>> b = dali.Tensor((2,3,4))
    >>> dali.hstack((a,b))
    Tensor([1, 2, 3, 2, 3, 4])
    >>> a = dali.Tensor([[1],[2],[3]])
    >>> b = dali.Tensor([[2],[3],[4]])
    >>> dali.hstack((a,b))
    Tensor([[1, 2],
            [2, 3],
            [3, 4]])
    """
    return Tensor.wrapc(c_hstack(ensure_tensor_list(tensors)))

cpdef Tensor vstack(tensors):
    """
    vstack(tensors)

    Arguments
    ---------
    tensors : a list or tuple of inputs of type Tensor

    Returns
    -------
    out : Tensor
        result of the reshape operation.
    """
    return Tensor.wrapc(c_vstack(ensure_tensor_list(tensors)))

cpdef Tensor gather(Tensor source, Tensor indices):
    """
    gather(source, indices)

    Arguments
    ---------
    source : Tensor location where gather is performed
    indices : Tensor containing the desired elements to be
              collected from the source.

    Returns
    -------
    out : Tensor
        result of the gather operation.
    """
    return Tensor.wrapc(c_gather(source.o, indices.o))
