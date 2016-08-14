from dali.tensor.tensor          cimport *
from dali.array.array            cimport *
from third_party.libcpp11.vector cimport vector
from ..array.dtype                      cimport *
import numpy                     as np
cimport third_party.modern_numpy as c_np

cdef extern from "dali/layers/layers.h" nogil:
    cdef cppclass CLayer "Layer":
        CTensor b
        CTensor W
        DType   dtype
        CDevice device

        int hidden_size
        int input_size

        CLayer()
        CLayer(int input_size, int hidden_size, DType dtype, CDevice device) except +
        CLayer(const CLayer&, bint copy_w, bint copy_dw) except +
        CTensor activate(CTensor) except+
        CLayer shallow_copy()
        vector[CTensor] parameters()

    cdef cppclass CStackedInputLayer "StackedInputLayer":
        CTensor b
        DType   dtype
        CDevice device

        vector[CTensor] tensors

        int hidden_size

        vector[CTensor] parameters()

        CStackedInputLayer()
        CStackedInputLayer(vector[int] input_size, int hidden_size, DType dtype, CDevice device) except +
        CStackedInputLayer(int input_size, int hidden_size, DType dtype, CDevice device) except +
        CStackedInputLayer(const CStackedInputLayer&, bint copy_w, bint copy_dw) except +

        const vector[int]& get_input_sizes()
        void set_input_sizes(vector[int] new_sizes) except +

        CTensor activate(CTensor) except+
        CTensor activate(const vector[CTensor]&) except+
        CTensor activate(CTensor, const vector[CTensor]&) except+

        CStackedInputLayer shallow_copy()

cdef class Layer:
    cdef CLayer o

    @staticmethod
    cdef Layer wrapc(CLayer o)

cdef class StackedInputLayer:
    cdef CStackedInputLayer o

    @staticmethod
    cdef StackedInputLayer wrapc(CStackedInputLayer o)

