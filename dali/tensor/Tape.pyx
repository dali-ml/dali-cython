from cpython.ref cimport PyObject

cdef extern from "dali/tensor/python_tape.h" nogil:
    void emplace_back(PyObject* callback)

cdef extern from "dali/tensor/Tape.h" namespace "graph" nogil:
    void backward()
    bint backprop_enabled()
    void clear()

class Graph:
    @staticmethod
    def emplace_back(backprop):
        cdef PyObject* backprop_ptr = (<PyObject*>backprop)
        with nogil:
            emplace_back(backprop_ptr)

    @staticmethod
    def backward():
        with nogil:
            backward()

    @staticmethod
    def clear():
        with nogil:
            clear()

    @staticmethod
    def backprop_enabled():
        return backprop_enabled()
