from cpython.ref cimport PyObject

cdef extern from "core/tensor/python_tape.h" nogil:
    void emplace_back(PyObject* callback)

cdef extern from "dali/tensor/Tape.h" namespace "graph" nogil:
    void backward() except+
    bint backprop_enabled()
    void clear()
    void _set_backprop_enabled(bint value);
    size_t size();

class Graph:
    @staticmethod
    def emplace_back(backprop):
        cdef PyObject* backprop_ptr = (<PyObject*>backprop)
        with nogil:
            emplace_back(backprop_ptr)

    @staticmethod
    def backward():
        backward()

    @staticmethod
    def clear():
        clear()

    @staticmethod
    def backprop_enabled():
        return backprop_enabled()

    @staticmethod
    def size():
        return size();


cdef class NoBackprop:
    cdef bint old_value
    cdef bint _enabled

    def __cinit__(NoBackprop self, enabled=True):
        self._enabled = enabled

    def __enter__(NoBackprop self):
        if self._enabled:
            self.old_value = Graph.backprop_enabled()
            _set_backprop_enabled(False)

    def __exit__(NoBackprop self, *args, **kwargs):
        if self._enabled:
            _set_backprop_enabled(self.old_value)

