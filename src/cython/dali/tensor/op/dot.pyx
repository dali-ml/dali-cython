cpdef Tensor dot(Tensor a, Tensor b):
    """
    dot(a, b)

    Dot product of two tensors.

    For 2-D tensors it is equivalent to matrix multiplication, and for 1-D
    tensors to inner product of vectors (without complex conjugation). For
    N dimensions it is a sum product over the last axis of `a` and
    the second-to-last of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D tensors then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    outer : compute the outer product of two vectors.
    tensordot : sum products over arbitrary axes.
    inner : inner product of two vectors
    matrixdot : ordinary matrix multiplication of 2D tensors
    matrix_vector_dot : matrix-vector multiplication
    """
    return Tensor.wrapc(cdot(a.o, b.o))

cpdef Tensor outer(Tensor a, Tensor b):
    """
    outer(a, b)

    Compute the outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::

      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) array_like
        First input vector.
    b : (N,) array_like
        Second input vector.

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    """
    return Tensor.wrapc(couter(a.o, b.o))

cpdef Tensor inner(Tensor a, Tensor b):
    """
    inner(a, b)

    Inner product of two tensors.

    Ordinary inner product of vectors for 1-D tensors.

    Parameters
    ----------
    a, b : Tensor
        `a` and `b` are 1-D tensors whose dimensions must match.

    Returns
    -------
    out : Tensor
        result of inner product of a, b. A scalar.

    Raises
    ------
    ValueError
        If the dimensions of `a` and `b` do not match.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    """
    return Tensor.wrapc(cinner(a.o, b.o))

cpdef Tensor matrixdot(Tensor a, Tensor b):
    """
    matrixdot(a, b)

    Ordinary matrix multiplication of two 2-D tensors.

    Parameters
    ----------
    a, b : Tensor
        `a` and `b` are 2-D tensors.

    Returns
    -------
    out : Tensor
        result of matrix multiplication of a, b. A 2-D Tensor.

    Raises
    ------
    ValueError
        If the second dimension of `a` and the first dimension of `b` does not match.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    """
    return Tensor.wrapc(cmatrixdot(a.o, b.o))

cpdef Tensor matrix_vector_dot(Tensor a, Tensor b):
    """
    matrix_vector_dot(a, b)

    Multiplication of a vector with a matrix.

    Parameters
    ----------
    a, b : Tensor
        One of `a` or `b` must be a 2-D Tensor, while the other a 1-D Tensor.

    Returns
    -------
    out : Tensor
        result of matrix-vector multiplication of a, b. A 1-D Tensor.

    Raises
    ------
    ValueError
        The last dimension of `a` must match the first dimension of `b`.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    """
    return Tensor.wrapc(cmatrix_vector_dot(a.o, b.o))

cdef bint is_int_list(object int_list):
    for val in int_list:
        if not isinstance(val, int):
            return False
    return True

cpdef Tensor tensordot(Tensor a, Tensor b, axes=2):
    """
    tensordot(a, b, axes=2)

    Compute tensor dot product along specified axes for tensors >= 1-D.

    Given two tensors (tensors of dimension greater than or equal to one),
    `a` and `b`, and an array_like object containing two array_like
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.

    Parameters
    ----------
    a, b : array_like, len(shape) >= 1
        Tensors to "dot".

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    See Also
    --------
    dot

    Notes
    -----
    Three common use cases are:
        ``axes = 0`` : tensor product $a\otimes b$
        ``axes = 1`` : tensor dot product $a\cdot b$
        ``axes = 2`` : (default) tensor double contraction $a:b$

    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    Examples
    --------
    A "traditional" example:

    >>> a = dali.arange(60.).reshape(3,4,5)
    >>> b = dali.arange(24.).reshape(4,3,2)
    >>> c = dali.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c.w
    Array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    """
    cdef vector[int] a_axes
    cdef vector[int] b_axes
    if isinstance(axes, int):
        return Tensor.wrapc(ctensordot(a.o, b.o, axes))
    elif isinstance(axes, (tuple, list)) and len(axes) == 2 and is_int_list(axes[0]) and is_int_list(axes[1]):
        a_axes = axes[0]
        b_axes = axes[1]
        return Tensor.wrapc(ctensordot(a.o, b.o, a_axes, b_axes))
    else:
        raise ValueError(
            "axes must be an integer or a tuple/list containing two "
            "int list/tuples of the same length (got " + str(axes) + ")."
        )
