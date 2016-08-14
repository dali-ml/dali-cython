from dali.tensor.tensor          cimport *
from dali.array.array            cimport *
from third_party.libcpp11.vector cimport vector
from ..array.dtype                      cimport *
from .layers                     cimport CStackedInputLayer, StackedInputLayer
import numpy                     as np
cimport third_party.modern_numpy as c_np

cdef extern from "dali/layers/gru.h" nogil:

    cdef cppclass CGRU "GRU":
        int hidden_size
        int input_size
        DType   dtype
        CDevice device

        CStackedInputLayer reset_layer
        CStackedInputLayer memory_interpolation_layer
        CStackedInputLayer memory_to_memory_layer

        CGRU() except +
        CGRU(int input_size, int hidden_size, DType, CDevice) except +
        CGRU(const CGRU&, bint copy_w, bint copy_dw) except +

        vector[CTensor] parameters()

        CTensor activate(CTensor) except+
        CGRU shallow_copy()
        CTensor initial_states()
        CTensor activate(CTensor input_vector, CTensor previous_state) except +
        CTensor activate_sequence(CTensor initial_state, vector[CTensor] input_sequence) except +

cdef class GRU:
    cdef CGRU o

    @staticmethod
    cdef GRU wrapc(CGRU o)
