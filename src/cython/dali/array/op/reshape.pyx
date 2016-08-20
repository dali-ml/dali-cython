cpdef AssignableArray concatenate(arrays, int axis):
    """
    concatenate(arrays)

    Arguments
    ---------
    arrays : a list or tuple of inputs of type Array
    axis : integer, axis along which to concatenate the arrays.

    Returns
    -------
    out : AssignableArray
        result of the reshape operation.
    """
    return AssignableArray.wrapc(c_concatenate(ensure_array_list(arrays), axis))

cpdef AssignableArray hstack(arrays):
    """
    hstack(arrays)

    Stack arrays in sequence horizontally (column wise).

    Take a sequence of arrays and stack them horizontally to make
    a single Array.

    Parameters
    ----------
    tup : sequence of arrays
        All arrays must have the same shape along all but the second axis.

    Returns
    -------
    stacked : AssignableArray
        The result of stacking the given arrays.

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
    >>> a = dali.Array((1,2,3))
    >>> b = dali.Array((2,3,4))
    >>> dali.hstack((a,b)).eval()
    Array([1, 2, 3, 2, 3, 4])
    >>> a = dali.Array([[1],[2],[3]])
    >>> b = dali.Array([[2],[3],[4]])
    >>> dali.hstack((a,b)).eval()
    Array([[1, 2],
           [2, 3],
           [3, 4]])
    """
    return AssignableArray.wrapc(c_hstack(ensure_array_list(arrays)))

cpdef AssignableArray vstack(arrays):
    """
    vstack(arrays)

    Arguments
    ---------
    arrays : a list or tuple of inputs of type Array

    Returns
    -------
    out : AssignableArray
        result of the reshape operation.
    """
    return AssignableArray.wrapc(c_vstack(ensure_array_list(arrays)))

cpdef AssignableArray gather(Array source, Array indices):
    """
    gather(source, indices)

    Arguments
    ---------
    source : Array location where gather is performed
    indices : Array containing the desired elements to be
              collected from the source.

    Returns
    -------
    out : AssignableArray
        result of the gather operation.
    """
    return AssignableArray.wrapc(c_gather(source.o, indices.o))

cpdef AssignableArray take_from_rows(Array source, Array indices):
    """
    take_from_rows(source, indices)

    Arguments
    ---------
    source : Array location where gather is performed on each row
    indices : Array containing the desired columns to be
              collected from the source's rows.

    Returns
    -------
    out : AssignableArray
        result of the take_from_rows operation.
    """
    return AssignableArray.wrapc(c_take_from_rows(source.o, indices.o))
