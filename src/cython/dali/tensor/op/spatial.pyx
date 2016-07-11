
cdef POOLING_T ensure_pooling(str pooling_mode):
    if pooling_mode == "MAX":
        return POOLING_T_MAX
    elif pooling_mode == "AVG":
        return POOLING_T_AVG
    else:
        raise ValueError(
            "pooling_mode must be equal to 'MAX' or 'AVG' (got " +
            str(pooling_mode) + ")."
        )

cdef PADDING_T ensure_padding(str padding):
    if padding == "VALID":
        return PADDING_T_VALID
    elif padding == "SAME":
        return PADDING_T_SAME
    else:
        raise ValueError(
            "padding must be equal to 'VALID' or 'SAME' (got " +
            str(padding) + ")."
        )

cpdef conv2d(Tensor t,
             Tensor filters,
             int stride_h,
             int stride_w,
             padding="VALID",
             data_format="NCHW"):
    """
    conv2d(t, filters, stride_h, stride_w, padding="VALID", data_format="NCHW")

    Return the 2D-convolution of the 4-dimensional tensor `t` with
    the contents of the 4-dimensional tensor `filters`.

    Parameters
    ----------

    t : Tensor, 4D input data for convolution. Dimensions correspond to
        those described by data_format argument (see-below).
    filters : Tensor, 4D input filters for convolution. Dimensions
              correspond to those described by data_format argument
              (see-below), with the letter 'N' corresponding to
              the dimension where the number of filters are defined.
    stride_h : integer, how much do filters move along the h-dimension
               each time they are applied.
    stride_w : integer, how much do filters move along the w-dimension
               each time they are applied.

    padding : string, if 'VALID' then the output is not padded and can be
              smaller than the input in spatial-dimensions;
              if 'SAME' then the input is padded with 0s in such a way that
              the output has the same spatial dimensions as the input
              (shape-preserving mode).
    data_format : a string description of the role of each dimension in the
                  input data and filters. Current supported options are:
                  'NCHW' and 'NHWC'. The letters have the following meaning:
                  - N : the batch-dimension/ number of filters dimension
                  - C : the input channels dimension
                  - H : the spatial h-dimension (e.g. height)
                  - W : the spatial w-dimension (e.g. weight)
                  Defaults to 'NCHW'. Note: tensorflow default is 'NHWC'.

    Returns
    -------

    out : Tensor
        result of convolving `t` by `filters`.
    """
    return Tensor.wrapc(
        c_conv2d(
            t.o,
            filters.o,
            stride_h,
            stride_w,
            ensure_padding(padding),
            data_format
        )
    )

cpdef max_pool(Tensor t,
               int window_h,
               int window_w,
               int stride_h,
               int stride_w,
               padding="VALID",
               data_format="NCHW"):
    """
    max_pool(t, window_h, window_w, stride_h, stride_w, padding="VALID", data_format="NCHW")

    Return the result of taking the spatial-maximum along both spatial
    dimensions h and w of the input 4-dimensional tensor `t` in each
    window of size (window_h, window_w) and sliding this window
    by (stride_h, stride_w) at each application.

    Parameters
    ----------

    t : Tensor, 4D input data for pooling. Dimensions correspond to
        those described by data_format argument (see-below).
    window_h : integer, size of the pooling in the h-dimension.
    window_w : integer, size of the pooling in the w-dimension.
    stride_h : integer, how much do filters move along the h-dimension
               each time they are applied.
    stride_w : integer, how much do filters move along the w-dimension
               each time they are applied.
    padding : string, if 'VALID' then the output is not padded and can be
              smaller than the input in spatial-dimensions;
              if 'SAME' then the input is padded with 0s in such a way that
              the output has the same spatial dimensions as the input
              (shape-preserving mode).
    data_format : a string description of the role of each dimension in the
                  input data and filters. Current supported options are:
                  'NCHW' and 'NHWC'. The letters have the following meaning:
                  - N : the batch-dimension
                  - C : the input channels dimension
                  - H : the spatial h-dimension (e.g. height)
                  - W : the spatial w-dimension (e.g. weight)
                  Defaults to 'NCHW'. Note: tensorflow default is 'NHWC'.

    Returns
    -------

    out : Tensor
        result of pooling `t`

    See Also
    --------
    pool2d : spatial pooling a 4D tensor
    avg_pool : spatial average-pooling a 4D tensor
    """
    return Tensor.wrapc(
        c_max_pool(
            t.o,
            window_h,
            window_w,
            stride_h,
            stride_w,
            ensure_padding(padding),
            data_format
        )
    )

cpdef avg_pool(Tensor t,
               int window_h,
               int window_w,
               int stride_h,
               int stride_w,
               padding="VALID",
               data_format="NCHW"):
    """
    avg_pool(t, window_h, window_w, stride_h, stride_w, padding="VALID", data_format="NCHW")

    Return the result of taking the spatial-average along both spatial
    dimensions h and w of the input 4-dimensional tensor `t` in each
    window of size (window_h, window_w) and sliding this window
    by (stride_h, stride_w) at each application.

    Parameters
    ----------

    t : Tensor, 4D input data for pooling. Dimensions correspond to
        those described by data_format argument (see-below).
    window_h : integer, size of the pooling in the h-dimension.
    window_w : integer, size of the pooling in the w-dimension.
    stride_h : integer, how much do filters move along the h-dimension
               each time they are applied.
    stride_w : integer, how much do filters move along the w-dimension
               each time they are applied.
    padding : string, if 'VALID' then the output is not padded and can be
              smaller than the input in spatial-dimensions;
              if 'SAME' then the input is padded with 0s in such a way that
              the output has the same spatial dimensions as the input
              (shape-preserving mode).
    data_format : a string description of the role of each dimension in the
                  input data and filters. Current supported options are:
                  'NCHW' and 'NHWC'. The letters have the following meaning:
                  - N : the batch-dimension
                  - C : the input channels dimension
                  - H : the spatial h-dimension (e.g. height)
                  - W : the spatial w-dimension (e.g. weight)
                  Defaults to 'NCHW'. Note: tensorflow default is 'NHWC'.

    Returns
    -------

    out : Tensor
        result of pooling `t`

    See Also
    --------
    pool2d : spatial pooling a 4D tensor
    max_pool : spatial maximum-pooling a 4D tensor
    """
    return Tensor.wrapc(
        c_avg_pool(
            t.o,
            window_h,
            window_w,
            stride_h,
            stride_w,
            ensure_padding(padding),
            data_format
        )
    )

cpdef pool2d(Tensor t,
             int window_h,
             int window_w,
             int stride_h,
             int stride_w,
             pooling_mode,
             padding="VALID",
             data_format="NCHW"):
    """
    pool2d(t, window_h, window_w, stride_h, stride_w, pooling_mode, padding="VALID", data_format="NCHW")

    Return the result of taking some summary pooling function along
    both spatial dimensions h and w of the input 4-dimensional tensor
    `t` in each window of size (window_h, window_w) and sliding this
    window by (stride_h, stride_w) at each application.

    Parameters
    ----------

    t : Tensor, 4D input data for pooling. Dimensions correspond to
        those described by data_format argument (see-below).
    window_h : integer, size of the pooling in the h-dimension.
    window_w : integer, size of the pooling in the w-dimension.
    stride_h : integer, how much do filters move along the h-dimension
               each time they are applied.
    stride_w : integer, how much do filters move along the w-dimension
               each time they are applied.
    pooling_mode : string, choices are 'MAX' for max-pooling and
                  'AVG' for average-pooling.
    padding : string, if 'VALID' then the output is not padded and can be
              smaller than the input in spatial-dimensions;
              if 'SAME' then the input is padded with 0s in such a way that
              the output has the same spatial dimensions as the input
              (shape-preserving mode).
    data_format : a string description of the role of each dimension in the
                  input data and filters. Current supported options are:
                  'NCHW' and 'NHWC'. The letters have the following meaning:
                  - N : the batch-dimension
                  - C : the input channels dimension
                  - H : the spatial h-dimension (e.g. height)
                  - W : the spatial w-dimension (e.g. weight)
                  Defaults to 'NCHW'. Note: tensorflow default is 'NHWC'.

    Returns
    -------

    out : Tensor
        result of pooling `t`

    See Also
    --------
    max_pool : spatial max-pooling a 4D tensor
    avg_pool : spatial average-pooling a 4D tensor
    """
    return Tensor.wrapc(
        c_pool2d(
            t.o,
            window_h,
            window_w,
            stride_h,
            stride_w,
            ensure_pooling(pooling_mode),
            ensure_padding(padding),
            data_format
        )
    )

cpdef im2col(Tensor t,
             int filter_h,
             int filter_w,
             int stride_h,
             int stride_w,
             data_format="NCHW"):
    """
    im2col(t, filter_h, filter_w, stride_h, stride_w, data_format="NCHW")

    Return a 2D-matrix that holds in each column a patch from the input
    tensor defined by the filter_h, filter_w and stride_h, stride_w
    parameters. The output of this operation is often used as input
    to a matrix multiply to perform convolutions using gemm instead
    of a more specialized operation.

    Parameters
    ----------

    t : Tensor, 4D input data for convolution. Dimensions correspond to
        those described by data_format argument (see-below).
    filter_h : integer, size of the patches in the h-dimension.
    filter_w : integer, size of the patches in the w-dimension.
    stride_h : integer, how much do filters move along the h-dimension
               each time they are applied.
    stride_w : integer, how much do filters move along the w-dimension
               each time they are applied.
    data_format : a string description of the role of each dimension in the
                  input data. Current supported options are:
                  'NCHW' and 'NHWC'. The letters have the following meaning:
                  - N : the batch-dimension
                  - C : the input channels dimension
                  - H : the spatial h-dimension (e.g. height)
                  - W : the spatial w-dimension (e.g. weight)
                  Defaults to 'NCHW'. Note: tensorflow default is 'NHWC'.

    Returns
    -------

    out : Tensor
        result of collecting patches in `t` organized in 2D.
    """
    return Tensor.wrapc(
        c_im2col(
            t.o,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            data_format
        )
    )

cpdef col2im(Tensor t,
             vector[int] image_shape,
             int filter_h,
             int filter_w,
             int stride_h,
             int stride_w,
             data_format="NCHW"):
    """
    col2im(t, image_shape, filter_h, filter_w, stride_h, stride_w, data_format="NCHW")

    Reduce the data contained in the 2D-tensor `t` and place back
    into a 4D Tensor (inverse operation of `im2col`). The output of
    this operation is often used to propagate gradients back through
    a gemm-based convolution.

    Parameters
    ----------

    t : Tensor, 2D input data typically resulting from im2col.
    image_shape : length-4 tuple/list of integers describing the original
                  image shape pre-im2col (e.g. output dimensions of
                  col2im).
    filter_h : integer, size of the patches in the h-dimension.
    filter_w : integer, size of the patches in the w-dimension.
    stride_h : integer, how much do filters move along the h-dimension
               each time they are applied.
    stride_w : integer, how much do filters move along the w-dimension
               each time they are applied.
    data_format : a string description of the role of each dimension in the
                  output data. Current supported options are:
                  'NCHW' and 'NHWC'. The letters have the following meaning:
                  - N : the batch-dimension/ number of filters dimension
                  - C : the input channels dimension
                  - H : the spatial h-dimension (e.g. height)
                  - W : the spatial w-dimension (e.g. weight)
                  Defaults to 'NCHW'. Note: tensorflow default is 'NHWC'.

    Returns
    -------

    out : Tensor
        result of reducing the data patches in `t` and placing the result
        in a 4D Tensor.
    """
    return Tensor.wrapc(
        c_col2im(
            t.o,
            image_shape,
            filter_h,
            filter_w,
            stride_h,
            stride_w,
            data_format
        )
    )

cpdef conv2d_add_bias(Tensor t,
                      Tensor bias,
                      data_format="NCHW"):
    """
    conv2d_add_bias(t, bias, data_format="NCHW")

    Add a 1D Tensor to a 4D Tensor while broadcasting the
    contents along each spatial and batch-dimension.
    Bias tensor must have the same length as the channel
    dimension of the tensor `t`.

    Parameters
    ----------

    t : Tensor, 4D input data for convolution. Dimensions correspond to
        those described by data_format argument (see-below).
    bias : Tensor, 1D bias vector.
    data_format : a string description of the role of each dimension in the
                  input data and filters. Current supported options are:
                  'NCHW' and 'NHWC'. The letters have the following meaning:
                  - N : the batch-dimension
                  - C : the input channels dimension
                  - H : the spatial h-dimension (e.g. height)
                  - W : the spatial w-dimension (e.g. weight)
                  Defaults to 'NCHW'. Note: tensorflow default is 'NHWC'.

    Returns
    -------

    out : Tensor
        result of adding `bias` to `t`
    """
    return Tensor.wrapc(
        c_conv2d_add_bias(
            t.o,
            bias.o,
            data_format
        )
    )
