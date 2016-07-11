from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector
from libcpp.string cimport string

cdef extern from "dali/array/op/spatial/spatial_enums.h":
    enum PADDING_T:
        PADDING_T_SAME, PADDING_T_VALID

    enum POOLING_T:
        POOLING_T_MAX, POOLING_T_AVG

cdef extern from "dali/tensor/op/spatial.h" namespace "tensor_ops":
    CTensor c_conv2d "tensor_ops::conv2d" (
        CTensor input,
        CTensor filters,
        int stride_h,
        int stride_w,
        PADDING_T padding,
        const string& data_format) except +

    CTensor c_im2col "tensor_ops::im2col" (
        CTensor input,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        const string& data_format) except +

    CTensor c_col2im "tensor_ops::col2im" (
        CTensor input,
        const vector[int]& image_shape,
        int filter_h,
        int filter_w,
        int stride_h,
        int stride_w,
        const string& data_format) except +

    CTensor c_conv2d_add_bias "tensor_ops::conv2d_add_bias" (
        CTensor conv_out,
        CTensor bias,
        const string& data_format) except +

    CTensor c_pool2d "tensor_ops::pool2d" (
        CTensor input,
        int window_h,
        int window_w,
        int stride_h,
        int stride_w,
        POOLING_T pooling_mode,
        PADDING_T padding,
        const string& data_format) except +

    CTensor c_max_pool "tensor_ops::max_pool" (
        CTensor input,
        int window_h,
        int window_w,
        int stride_h,
        int stride_w,
        PADDING_T padding,
        const string& data_format) except +

    CTensor c_avg_pool "tensor_ops::avg_pool" (
        CTensor input,
        int window_h,
        int window_w,
        int stride_h,
        int stride_w,
        PADDING_T padding,
        const string& data_format) except +
