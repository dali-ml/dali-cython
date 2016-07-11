def emplace_back(backprop):
    """
    emplace_back(backprop)

    Add a python method that takes no arguments
    to the computation tape so that it is called
    when performing backpropagation.

    Parameters
    ----------
    backprop : a callable Python function that takes
               no arguments.

    See Also
    --------
    backward : perform backpropagation.
    """
    cdef PyObject* backprop_ptr = (<PyObject*>backprop)
    with nogil:
        c_emplace_back(backprop_ptr)

def backward():
    """
    backward()

    Perform backpropagation of the currently set
    objective function and save gradients into
    each involved Tensor's dw Array. Clear
    the computation tape.
    """
    c_backward()

def clear():
    """
    clear()

    Erase the operations and temporary storage
    needed for backpropagation.
    """
    c_clear()

def backprop_enabled():
    """
    backprop_enabled()

    Check whether Tensor operations are being
    recorded for backpropagation.

    Returns
    -------

    enabled : bool
        whether backpropagation recording is on
    """
    return c_backprop_enabled()

def size():
    """
    size()

    Return the number of operations currently
    stored on the computation tape for
    backpropagation.

    Returns
    -------

    tape_length : int
        how many operations will be run during backpropagation
    """
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

