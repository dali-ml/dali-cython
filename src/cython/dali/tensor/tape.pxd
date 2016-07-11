from cpython.ref cimport PyObject

cdef extern from "python_tape.h" nogil:
    void c_emplace_back "emplace_back"(PyObject* callback)

cdef extern from "dali/tensor/tape.h" namespace "graph" nogil:
    void c_backward "graph::backward"() except+
    bint c_backprop_enabled "graph::backprop_enabled"()
    void c_clear "graph::clear"()
    void c__set_backprop_enabled "graph::_set_backprop_enabled"(bint value);
    size_t c_size "graph::size"();

cdef class NoBackprop:
    """
    NoBackprop()

    Control whether a code-segment's tensor
    operations will be used to compute gradients.

    Usage
    -----
    In this example the tensors a and b
    receive no gradient from c, even though
    c was added to the objective function
    using `grad()`:

    ```
    a = dali.Tensor((2, 3))
    b = dali.Tensor((2, 3))
    with NoBackprop():
        c = a + b
        c.grad()
        backward()
    ```
    """
    cdef bint old_value
    cdef bint _enabled
