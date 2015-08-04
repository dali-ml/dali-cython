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

# cdef class NoBackprop():
#     bint old_value
#     bint _enabled

#     def __cinit__(NoBackprop self, enabled=True)
#         self._enabled = enabled

#     def __enter__(NoBackprop self):
#         if self._enabled:
#             self.old_value = Graph.backprop_enabled()
#             Graph.set_backprop(True)

#     def __exit(NoBackprop self, *args, **kwargs):
#         if self._enabled:
#             Graph.set_backprop(self.old_value)

