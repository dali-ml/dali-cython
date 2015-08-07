from libcpp.string   cimport string
from libcpp11.vector cimport vector
from libcpp11.memory cimport shared_ptr
# Import the Python-level symbols of numpy
import numpy as np
# Import the C-level symbols of numpy
cimport modern_numpy as np

cdef string normalize_s(s):
    if type(s) is str:
        return s.encode("utf-8")
    elif type(s) is bytes:
        return s
    else:
        raise TypeError("Must pass a str or bytes object.")

cdef bint is_fdtype(np.NPY_TYPES type_id) nogil:
    return type_id == np.NPY_FLOAT32 or \
           type_id == np.NPY_FLOAT64

cdef inline void ensure_fdtype(np.NPY_TYPES type_id):
    if not is_fdtype(type_id):
        raise ValueError(
            "Invalid dtype: " +
            str(np.PyArray_DescrFromType(type_id)) +
            " (should be one of float32, float64)")

# File IO, save / load, etc...
include "dali/utils/core_utils.pyx"

# Used for storing Tensor buffers
include "dali/math/TensorInternal.pyx"

# Matrix class
include "dali/tensor/Mat.pyx"

# Matrix initialization with random numbers.
include "dali/tensor/random.pyx"

# Softmax, crossentropy etc....
include "dali/tensor/MatOps.pyx"

# Related to backpropagation.
include "dali/tensor/Tape.pyx"

# Layer, RNN, StackedInputLayer, etc...
include "dali/layers/Layers.pyx"

include "dali/layers/GRU.pyx"

include "dali/layers/LSTM.pyx"

# # Matrix class
include "dali/data_processing/Batch.pyx"

# # State for StackedModel and StackedGatedModel
include "dali/models/StackedModelState.pyx"

# # Stacked Model, a stacked LSTM with embedding and
# # decoder all in one.
include "dali/models/StackedModel.pyx"

# SGD, Adagrad, Adadelta, etc...
include "dali/tensor/Solver.pyx"

