from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector
from third_party.libcpp11.string cimport string

cdef extern from "dali/tensor/op/spatial.h" namespace "tensor_ops":
    CTensor conv2d(CTensor input,
                   CTensor filters,
                   int stride_h,
                   int stride_w,
                   PADDING_T padding,
                   const string& data_format)

    CTensor im2col(CTensor input,
                   int filter_h,
                   int filter_w,
                   int stride_h,
                   int stride_w,
                   const string& data_format)

    CTensor col2im(CTensor input,
                   const vector[int]& image_shape,
                   int filter_h,
                   int filter_w,
                   int stride_h,
                   int stride_w,
                   const string& data_format)

    CTensor conv2d_add_bias(CTensor conv_out,
                            CTensor bias,
                            const string& data_format)

    CTensor pool2d(CTensor input,
                   int window_h,
                   int window_w,
                   int stride_h,
                   int stride_w,
                   POOLING_T pooling_mode,
                   PADDING_T padding,
                   const string& data_format)

    CTensor max_pool(CTensor input,
                     int window_h,
                     int window_w,
                     int stride_h,
                     int stride_w,
                     PADDING_T padding,
                     const string& data_format)

    CTensor avg_pool(CTensor input,
                     int window_h,
                     int window_w,
                     int stride_h,
                     int stride_w,
                     PADDING_T padding,
                     const string& data_format)
