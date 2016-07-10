from cpython.ref cimport PyObject

cdef extern from "python_tape.h" nogil:
    void c_emplace_back "emplace_back"(PyObject* callback)

cdef extern from "dali/tensor/tape.h" namespace "graph" nogil:
    void c_backward "graph::backward"() except+
    bint c_backprop_enabled "graph::backprop_enabled"()
    void c_clear "graph::clear"()
    void c__set_backprop_enabled "graph::_set_backprop_enabled"(bint value);
    size_t c_size "graph::size"();

cdef class Graph:
    pass

cdef class NoBackprop:
    cdef bint old_value
    cdef bint _enabled
