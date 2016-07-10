from ..tensor                    cimport CTensor, Tensor
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/tensor/op/binary.h" namespace "tensor_ops":
	CTensor add(const CTensor&, const CTensor&);
    CTensor add(const vector[CTensor]& tensors);
    CTensor sub(const CTensor&, const CTensor&);
    CTensor eltmul(const CTensor&, const CTensor&);
    CTensor eltdiv(const CTensor&, const CTensor&);
    CTensor pow(const CTensor&, const CTensor& exponent);
    CTensor circular_convolution(const CTensor& content, const CTensor& shift);
    CTensor prelu(const CTensor& x, const CTensor& weights);
