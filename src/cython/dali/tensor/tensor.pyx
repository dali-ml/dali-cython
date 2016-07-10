import numpy as np
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
c_np.import_array()

cdef list_from_args(object args):
    if len(args) > 0:
        if all([isinstance(arg, int) for arg in args]):
            return args
        if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
            return args[0]
    raise ValueError("expected a list of integers")

cpdef Tensor ensure_tensor(object arr):
    if type(arr) == Tensor:
        return arr
    else:
        raise ValueError("not implemented")

cdef vector[CTensor] ensure_tensor_list(object tensors):
    cdef vector[CTensor] tensors_c
    cdef Tensor tensor_c

    got_list_of_tensors = (
        isinstance(tensors, (tuple, list)) and
        all([type(t) == Tensor for t in tensors])
    )

    if not got_list_of_tensors:
        raise ValueError("expected a list or a tuple of tensors")

    for tensor in tensors:
        tensor_c = tensor
        tensors_c.emplace_back(tensor_c.o)

    return tensors_c

cdef class Tensor:
    def __cinit__(Tensor self, vector[int] shape, dtype=np.float32, preferred_device=None):
        cdef Device device
        device = ensure_device(preferred_device)
        self.o = CTensor(shape, dtype_np_to_dali(dtype), device.o)

    @staticmethod
    cdef Tensor wrapc(CTensor o):
        ret = Tensor([])
        ret.o = o
        return ret

    cdef c_np.NPY_TYPES cdtype(Tensor self):
        return dtype_dali_to_c_np(self.o.dtype())

    property dtype:
        def __get__(Tensor self):
            return dtype_dali_to_np(self.o.dtype())

    property w:
        def __get__(Tensor self):
            return Array.wrapc(self.o.w)

    property dw:
        def __get__(Tensor self):
            return Array.wrapc(self.o.dw)

    property shape:
        def __get__(Tensor self):
            return tuple(self.o.shape())

    property preferred_device:
        def __get__(Tensor self):
            return Device.wrapc(self.o.preferred_device())

    property number_of_elements:
        def __get__(Array self):
            return self.o.number_of_elements()

    property size:
        def __get__(Array self):
            return self.o.number_of_elements()

    property ndim:
        def __get__(Array self):
            return self.o.ndim()

    def swapaxes(Array self, int axis1, int axis2):
        """a.swapaxes(axis1, axis2)

        Return a view of the tensor with `axis1` and `axis2` interchanged.

        Refer to `dali.swapaxes` for full documentation.

        See Also
        --------
        dali.swapaxes : equivalent function
        """
        return Tensor.wrapc(self.o.swapaxes(axis1, axis2))

    def ravel(Tensor self):
        """a.ravel()

        Return a flattened array.
        Note: if the memory in `a` cannot be reshaped without performing
        a copy, a copy is performed automatically to permit the
        shape transformation.

        Refer to `dali.ravel` for full documentation.

        See Also
        --------
        dali.ravel : equivalent function
        """
        return Tensor.wrapc(self.o.ravel())

    def copyless_ravel(Tensor self):
        """a.copyless_ravel()

        Return a flattened Tensor.
        Will raise an error if the data in `a` cannot be reshaped
        to 1D without performing a copy (e.g. due to strides that
        are irregular and prevent dimensions from being collapsed
        together).

        Refer to `dali.ravel` for full documentation.

        See Also
        --------
        dali.ravel : equivalent function
        """
        return Tensor.wrapc(self.o.copyless_ravel())

    def reshape(Tensor self, *newshape):
        """a.reshape(shape)

        Returns a Tensor containing the same data with a new shape.
        Note: if the memory in `a` cannot be reshaped without performing
        a copy, a copy is performed automatically to permit the
        shape transformation.

        Refer to `dali.reshape` for full documentation.

        See Also
        --------
        dali.reshape : equivalent function
        Tensor.copyless_reshape : equivalent function, raises error on copy
        dali.copyless_reshape : equivalent function, raises error on copy
        """
        cdef vector[int] cnewshape = list_from_args(newshape)
        return Tensor.wrapc(self.o.reshape(cnewshape))

    def copyless_reshape(Tensor self, *newshape):
        """a.copyless_reshape(shape)

        Returns a Tensor containing the same data with a new shape.
        Will raise an error if the data in `a` cannot be reshaped
        to the newshape without performing a copy (e.g. due
        to strides that are irregular and prevent dimensions
        from being collapsed together).

        Refer to `dali.copyless_reshape` for full documentation.

        See Also
        --------
        Tensor.reshape : equivalent function, will not raise error if copy required
        dali.reshape : equivalent function, will not raise error if copy required
        dali.copyless_reshape : equivalent function
        """
        cdef vector[int] cnewshape = list_from_args(newshape)
        return Tensor.wrapc(self.o.reshape(cnewshape))

    @staticmethod
    def zeros(vector[int] shape, dtype=np.float32, preferred_device=None):
        """zeros(shape, dtype=np.float32, preferred_device=None)

        Returns a Tensor filled with zeros.

        Parameters
        ----------
        shape: [int]
            a list representing sizes of subsequent dimensions
        dtype: np.dtype
            datatype used for representing numbers
        preferred_device: dali.Device
            preferred device for data storage. If it is equal to None,
            a dali.default_device() is used.
        """
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.zeros(shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def empty(vector[int] shape, dtype=np.float32, preferred_device=None):
        """empty(shape, dtype=np.float32, preferred_device=None)

        Returns a Tensor with uninitialized values.

        Parameters
        ----------
        shape: [int]
            a list representing sizes of subsequent dimensions
        dtype: np.dtype
            datatype used for representing numbers
        preferred_device: dali.Device
            preferred device for data storage. If it is equal to None,
            a dali.default_device() is used.
        """
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor(shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def ones(vector[int] shape, dtype=np.float32, preferred_device=None):
        """ones(shape, dtype=np.float32, preferred_device=None)

        Returns a Tensor filled with ones.

        Parameters
        ----------
        shape: [int]
            a list representing sizes of subsequent dimensions
        dtype: np.dtype
            datatype used for representing numbers
        preferred_device: dali.Device
            preferred device for data storage. If it is equal to None,
            a dali.default_device() is used.
        """
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.ones(shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def arange(start=None, stop=None, step=None, shape=None, dtype=np.float32, preferred_device=None):
        """
        arange([start,] stop[, step,], shape=None, dtype=np.float32, preferred_device=None)

        Return evenly spaced values within a given interval.

        Values are generated within the half-open interval ``[start, stop)``
        (in other words, the interval including `start` but excluding `stop`).
        For integer arguments the function is equivalent to the Python built-in
        `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
        but returns an Tensor rather than a list.

        When using a non-integer step, such as 0.1, the results will often not
        be consistent.  It is better to use ``linspace`` for these cases.

        Parameters
        ----------
        start : number, optional
            Start of interval.  The interval includes this value.  The default
            start value is 0.
        stop : number
            End of interval.  The interval does not include this value, except
            in some cases where `step` is not an integer and floating point
            round-off affects the length of `out`.
        step : number, optional
            Spacing between values.  For any output `out`, this is the distance
            between two adjacent values, ``out[i+1] - out[i]``.  The default
            step size is 1.  If `step` is specified, `start` must also be given.
        shape: [int]
            a list representing sizes of subsequent dimensions.
            Note: if start, stop, step are omitted, then shape can be used
            to simulate
            ``arange(start=0, stop=product(shape), step=1).reshape(shape)``
        dtype: np.dtype
            datatype used for representing numbers
        preferred_device: dali.Device
            preferred device for data storage. If it is equal to None,
            a dali.default_device() is used.

        Returns
        -------
        arange : Tensor
            Tensor of evenly spaced values.

            For floating point arguments, the length of the result is
            ``ceil((stop - start)/step)``.  Because of floating point overflow,
            this rule may result in the last element of `out` being greater
            than `stop`.
        """
        cdef Device device = ensure_device(preferred_device)
        cdef CTensor out_c
        if start is None and stop is None and shape is None:
            raise ValueError("start, stop, shape cannot all be None.")

        if start is None and stop is None and step is None and shape is not None:
            return Tensor.wrapc(CTensor.arange(shape, dtype_np_to_dali(dtype), device.o))

        if start is not None and stop is None:
            stop = start
            start = 0.0

        if step is None:
            step = 1.0

        if start is None:
            start = 0.0

        if shape is None:
            return Tensor.wrapc(CTensor.arange_double(start, stop, step, dtype_np_to_dali(dtype), device.o))
        else:
            out_c = CTensor.arange_double(start, stop, step, dtype_np_to_dali(dtype), device.o)
            out_c = out_c.reshape(shape)
            return Tensor.wrapc(out_c)

    @staticmethod
    def gaussian(double mean, double std, vector[int] shape, dtype=np.float32, preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.gaussian(mean, std, shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def uniform(double low, double high, vector[int] shape, dtype=np.float32, preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.uniform(low, high, shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def bernoulli(double prob, vector[int] shape, dtype=np.float32, preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.bernoulli(prob, shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def bernoulli_normalized(double prob, vector[int] shape, dtype=np.float32, preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.bernoulli_normalized(prob, shape, dtype_np_to_dali(dtype), device.o))
