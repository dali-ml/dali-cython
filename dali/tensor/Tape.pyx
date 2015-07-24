from cpython.ref cimport PyObject

cdef extern from "dali/tensor/python_tape.h":
    void emplace_back(PyObject* callback)

cdef extern from "dali/tensor/Tape.h" namespace "graph":
    void backward()
    bint backprop_enabled();

class Graph:
    @staticmethod
    def emplace_back(backprop):
        cdef PyObject* backprop_ptr = (<PyObject*>backprop)
        emplace_back(backprop_ptr)

    @staticmethod
    def backward():
        backward()

    @staticmethod
    def backprop_enabled():
        return backprop_enabled()
