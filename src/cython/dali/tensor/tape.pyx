def emplace_back(backprop):
    cdef PyObject* backprop_ptr = (<PyObject*>backprop)
    with nogil:
        c_emplace_back(backprop_ptr)

def backward():
    c_backward()

def clear():
    c_clear()

def backprop_enabled():
    return c_backprop_enabled()

def size():
    return c_size()


cdef class NoBackprop:
    def __cinit__(NoBackprop self, enabled=True):
        self._enabled = enabled

    def __enter__(NoBackprop self):
        if self._enabled:
            self.old_value = backprop_enabled()
            c__set_backprop_enabled(False)

    def __exit__(NoBackprop self, *args, **kwargs):
        if self._enabled:
            c__set_backprop_enabled(self.old_value)

