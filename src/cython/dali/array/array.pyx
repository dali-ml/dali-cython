import numpy as np

from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
c_np.import_array()

cdef object list_from_args(object args):
    if len(args) > 0:
        if all(isinstance(arg, int) for arg in args):
            return args
        if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
            return args[0]
    raise ValueError("expected a list of integers")


cdef class DoNotInitialize:
    """For internal use only - it exists so that Array initialization
    can be deferred to C++ functions."""
    pass

cpdef Array ensure_array(object arr):
    if type(arr) == Array:
        return arr
    else:
        return Array(arr, borrow=True)

cdef class AssignableArray:
    @staticmethod
    cdef AssignableArray wrapc(CAssignableArray o) except +:
        ret = AssignableArray()
        ret.o = o
        return ret

    def __cinit__(AssignableArray self):
        pass

    cpdef Array eval(AssignableArray self):
        return Array.wrapc(self.o.eval())

cdef class Array:
    def __cinit__(Array self, object data, dtype=None, preferred_device=None, borrow=False):
        if type(data) == DoNotInitialize:
            return
        assert isinstance(data, (list, tuple, np.ndarray)), \
                "dali only knows how to construct Arrays from list, tuple or np.array, " + \
                "got object of type " + str(type(data))

        cdef c_np.ndarray np_data
        if isinstance(data, np.ndarray):
            np_data = data

        if isinstance(data, (list, tuple)):
            np_data = np.array(data)
            # we are sure that numpy makes a copy here, so
            # we can safely steal the memory.
            borrow = True

        cdef Device device = ensure_device(preferred_device)

        cdef c_np.NPY_TYPES c_np_dtype = c_np.NPY_NOTYPE
        if dtype is not None:
            c_np_dtype = c_np.dtype(dtype).num

        self.use_numpy_memory(np_data, c_np_dtype, device.o, borrow)

    cdef void use_numpy_memory(Array self,
                               c_np.ndarray py_array,
                               c_np.NPY_TYPES dtype,
                               CDevice preferred_device,
                               bint steal) except +:
        if is_cnp_dtype_supported(py_array.dtype.num):
            if dtype == c_np.NPY_NOTYPE:
                dtype = py_array.dtype.num
            else:
                if dtype != py_array.dtype.num:
                    py_array = py_array.astype(np.PyArray_DescrFromType(dtype))
                    steal = True
        else:
            if dtype == c_np.NPY_NOTYPE:
                if np.issubdtype(py_array.dtype, np.float):
                    py_array = py_array.astype(np.float32)
                    dtype = py_array.dtype.num
                    steal = True
                elif np.issubdtype(py_array.dtype, np.integer):
                    py_array = py_array.astype(np.int32)
                    dtype = py_array.dtype.num
                    steal = True
                else:
                    raise ValueError("Cannot create Array from type " +
                                     str(py_array.dtype) +
                                     " (should be integer or floating point)")
            else:
                if dtype != py_array.dtype.num:
                    py_array = py_array.astype(c_np.PyArray_DescrFromType(dtype))

        cdef vector[int] shape
        shape.assign(c_np.PyArray_DIMS(py_array),
                     c_np.PyArray_DIMS(py_array) + c_np.PyArray_NDIM(py_array))
        cdef vector[int] strides
        strides.assign(c_np.PyArray_STRIDES(py_array),
                       c_np.PyArray_STRIDES(py_array) + c_np.PyArray_NDIM(py_array))

        cdef DType dali_dtype = dtype_c_np_to_dali(dtype)

        cdef int idx
        for idx in range(strides.size()):
            strides[idx] /= dtype_to_itemsize(dali_dtype)

        cdef CArray wrapper = CArray.adopt_buffer(c_np.PyArray_DATA(py_array),
                                                  shape,
                                                  dali_dtype,
                                                  CDevice.cpu(),
                                                  strides)

        if steal and py_array.flags.owndata and py_array.flags.writeable:
            self.o = wrapper
            c_np.PyArray_CLEARFLAGS(py_array, c_np.NPY_OWNDATA)
            self.o.to_device(preferred_device)
        else:
            self.o = CArray(shape, dali_dtype, preferred_device)
            self.o.copy_from(wrapper)

    @staticmethod
    cdef Array wrapc(CArray o) except +:
        ret = Array(DoNotInitialize())
        ret.o = o
        return ret

    cdef c_np.NPY_TYPES cdtype(Array self) except +:
        return dtype_dali_to_c_np(self.o.dtype())

    property offset:
        def __get__(Array self):
            return self.o.offset()

    @staticmethod
    def zeros(vector[int] shape, dtype=np.float32, preferred_device=None):
        """zeros(shape, dtype=np.float32, preferred_device=None)

        Returns an Array filled with zeros.

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
        return Array.wrapc(CArray.zeros(shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def arange(start=None, stop=None, step=None, shape=None, dtype=np.float32, preferred_device=None):
        """
        arange([start,] stop[, step,], shape=None, dtype=np.float32, preferred_device=None)

        Return evenly spaced values within a given interval.

        Values are generated within the half-open interval ``[start, stop)``
        (in other words, the interval including `start` but excluding `stop`).
        For integer arguments the function is equivalent to the Python built-in
        `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
        but returns an Array rather than a list.

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
        arange : Array
            Array of evenly spaced values.

            For floating point arguments, the length of the result is
            ``ceil((stop - start)/step)``.  Because of floating point overflow,
            this rule may result in the last element of `out` being greater
            than `stop`.
        """
        cdef Device device = ensure_device(preferred_device)
        cdef CArray out_c
        if start is None and stop is None and shape is None:
            raise ValueError("start, stop, shape cannot all be None.")

        if start is None and stop is None and step is None and shape is not None:
            return Array.wrapc(CArray.arange(shape, dtype_np_to_dali(dtype), device.o))

        if start is not None and stop is None:
            stop = start
            start = 0.0

        if step is None:
            step = 1.0

        if start is None:
            start = 0.0

        if shape is None:
            return Array.wrapc(CArray.arange_double(start, stop, step, dtype_np_to_dali(dtype), device.o))
        else:
            out_c = CArray.arange_double(start, stop, step, dtype_np_to_dali(dtype), device.o)
            out_c = out_c.reshape(shape)
            return Array.wrapc(out_c)

    @staticmethod
    def empty(vector[int] shape, dtype=np.float32, preferred_device=None):
        """empty(shape, dtype=np.float32, preferred_device=None)

        Returns an Array with uninitialized values.

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
        return Array.wrapc(CArray(shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def ones(vector[int] shape, dtype=np.float32, preferred_device=None):
        """ones(shape, dtype=np.float32, preferred_device=None)

        Returns an Array filled with ones.

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
        return Array.wrapc(CArray.ones(shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def zeros_like(other):
        cdef Array a = ensure_array(other)
        return Array.wrapc(CArray.zeros_like(a.o))

    @staticmethod
    def empty_like(other):
        cdef Array a = ensure_array(other)
        return Array.wrapc(CArray.empty_like(a.o))

    @staticmethod
    def ones_like(other):
        cdef Array a = ensure_array(other)
        return Array.wrapc(CArray.ones_like(a.o))

    property dtype:
        def __get__(Array self):
            return dtype_dali_to_np(self.o.dtype())

    property shape:
        def __get__(Array self):
            return tuple(self.o.shape())

    property bshape:
        def __get__(Array self):
            return tuple(self.o.bshape())

    property is_transpose:
        def __get__(Array self):
            return self.o.is_transpose()

    property spans_entire_memory:
        def __get__(Array self):
            return self.o.spans_entire_memory()

    property contiguous_memory:
        def __get__(Array self):
            return self.o.contiguous_memory()

    property number_of_elements:
        def __get__(Array self):
            return self.o.number_of_elements()

    property size:
        def __get__(Array self):
            return self.o.number_of_elements()

    property ndim:
        def __get__(Array self):
            return self.o.ndim()

    def __getitem__(Array self, args):
        if not isinstance(args, (tuple, list)):
            args = [args]

        cdef int arg_as_int
        cdef CBroadcast br

        cdef bint                    use_array = True
        cdef CArray                  arr       = self.o
        cdef CSlicingInProgressArray slicing

        for arg in args:
            if type(arg) == slice:
                if use_array:
                    slicing   = arr.operator_bracket(parse_slice(arg))
                    use_array = False
                else:
                    slicing = slicing.operator_bracket(parse_slice(arg))
            elif type(arg) == int:
                arg_as_int = arg
                if use_array:
                    arr = arr.operator_bracket(arg_as_int)
                else:
                    slicing = slicing.operator_bracket(arg_as_int)
            elif arg is None:
                if use_array:
                    slicing = arr.operator_bracket(br)
                    use_array = False
                else:
                    slicing = slicing.operator_bracket(br)
            else:
                raise TypeError("Cannot index array by object of type " + type(arg))
        if use_array:
            return Array.wrapc(arr)
        else:
            return Array.wrapc(slicing.toarray())

    def clear(Array self):
        """a.clear()

        Inplace operation that replaces all the contents of the array with zeros.
        """
        self.o.clear()

    def subshape(Array self):
        """a.subshape()

        Return the shape of a subtensor of this array,
        and is equivalent to `a[0].shape`.
        If `a` is a scalar, this method returns an empty
        tuple.
        """
        return tuple(self.o.subshape())

    property preferred_device:
        def __get__(Array self):
            return Device.wrapc(self.o.preferred_device())

    property strides:
        def __get__(Array self):
            return self.o.normalized_strides()

    def to_device(Array self, Device dev):
        self.o.to_device(dev.o)

    property T:
        def __get__(Array self):
            return self.transpose()

    def transpose(Array self, *axes):
        """a.transpose(*axes)

        Returns a view of the array with axes transposed.

        For a 1-D array, this has no effect.
        For a 2-D array, this is the usual matrix transpose.
        For an n-D array, if axes are given, their order indicates how the
        axes are permuted. If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : omitted, tuple of ints, or `n` ints

         * no argument: reverses the order of the axes.

         * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
           `i`-th axis becomes `a.transpose()`'s `j`-th axis.

         * `n` ints: same as an n-tuple of the same ints (this form is
           intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : Array
            View of `a`, with axes suitably permuted.

        See Also
        --------
        Array.T : Array property returning the array transposed.
        """
        cdef vector[int] cdims
        if len(dims) == 0:
            return Array.wrapc(self.o.transpose())
        else:
            caxes = list_from_args(axes)
            return Array.wrapc(self.o.transpose(caxes))

    def squeeze(Array self, axis=None):
        """a.squeeze(axis=None)

        Remove single-dimensional entries from the shape of an array.

        Parameters
        ----------
        a : array_like
            Input data.
        axis : None or int or tuple of ints, optional

            Selects a subset of the single-dimensional entries in the
            shape. If an axis is selected with shape entry greater than
            one, an error is raised.

        Returns
        -------
        squeezed : Array
            The input array, but with all or a subset of the
            dimensions of length 1 removed. This is always `a` itself
            or a view into `a`.
        """
        if axis is None:
            shape = [dim for dim in self.shape if dim != 1]
            return self.copyless_reshape(shape)
        elif isinstance(axis, int):
            return Array.wrapc(self.o.squeeze(axis))
        elif isinstance(axis, (tuple, list)):
            newshape = list(self.shape)
            for dim in axis:
                if dim < 0:
                    dim = len(newshape) + dim
                if dim < 0 or dim >= len(newshape):
                    raise ValueError(
                        "squeeze dimension (" + str(dim) +
                        ") must be less the dimensionality of compacted tensor (" +
                        str(len(newshape)) + ")."
                    )
                if newshape[dim] == 1:
                    newshape[dim] = None
                else:
                    raise ValueError(
                        "cannot select an axis to squeeze out which has size not equal to one (got axis=" +
                        str(dim) + ", shape[" + str(dim) + "]=" + str(newshape[dim]) + ").")
            newshape = [dimension for dimension in newshape if dimension is not None]
            return self.copyless_reshape(newshape)
        else:
            raise TypeError(
                "None, an integer, or a tuple of integers is "
                "required as argument to squeeze."
            )

    def swapaxes(Array self, int axis1, int axis2):
        """a.swapaxes(axis1, axis2)

        Return a view of the array with `axis1` and `axis2` interchanged.

        Refer to `dali.swapaxes` for full documentation.

        See Also
        --------
        dali.swapaxes : equivalent function
        """
        return Array.wrapc(self.o.swapaxes(axis1, axis2))

    def get_value(self, copy=False):
        """a.get_value(copy=False)

        Return a numpy array containing the same data as contained in `a`.
        The copy flag controls whether the numpy array is a view onto
        `a`'s memory, or whether it should allocate a different
        array altogether and replicate `a` inside the numpy array.
        """

        if copy:
            return np.array(self.get_value(False), copy=True)
        cdef c_np.ndarray ndarray
        cdef c_np.npy_intp* np_shape = <c_np.npy_intp*>malloc(self.o.ndim() * sizeof(c_np.npy_intp))
        cdef vector[int] arr_shape = self.o.shape()
        cdef vector[int] strides = self.o.normalized_strides()

        try:
            for i in range(self.o.ndim()):
                np_shape[i] = <c_np.npy_intp>(arr_shape[i])
            ndarray = c_np.PyArray_SimpleNewFromData(
                self.o.ndim(),
                np_shape,
                self.cdtype(),
                self.o.memory().get()[0].mutable_data(CDevice.cpu())
            )

            for i in range(self.o.ndim()):
                c_np.PyArray_STRIDES(ndarray)[i] = strides[i] * dtype_to_itemsize(self.o.dtype())
        finally:
            free(np_shape)
        ndarray.base = <PyObject*> self
        Py_INCREF(self)
        return ndarray

    def ravel(Array self):
        """a.ravel()

        Return a flattened array.
        Note: if the memory in `a` cannot be reshaped without performing
        a copy, a copy is performed automatically to permit the
        shape transformation.

        Refer to `dali.ravel` for full documentation.

        See Also
        --------
        dali.ravel : equivalent function
        Array.flatten : a 1D copy of the array.
        """
        return Array.wrapc(self.o.ravel())

    def copyless_ravel(Array self):
        """a.copyless_ravel()

        Return a flattened array.
        Will raise an error if the data in `a` cannot be reshaped
        to 1D without performing a copy (e.g. due to strides that
        are irregular and prevent dimensions from being collapsed
        together).

        Refer to `dali.ravel` for full documentation.

        See Also
        --------
        dali.ravel : equivalent function
        Array.flatten : a 1D copy of the array.
        """
        return Array.wrapc(self.o.copyless_ravel())

    def reshape(Array self, *newshape):
        """a.reshape(shape)

        Returns an array containing the same data with a new shape.
        Note: if the memory in `a` cannot be reshaped without performing
        a copy, a copy is performed automatically to permit the
        shape transformation.

        Refer to `dali.reshape` for full documentation.

        See Also
        --------
        dali.reshape : equivalent function
        Array.copyless_reshape : equivalent function, raises error on copy
        dali.copyless_reshape : equivalent function, raises error on copy
        """
        cdef vector[int] cnewshape = list_from_args(newshape)
        return Array.wrapc(self.o.reshape(cnewshape))

    def copyless_reshape(Array self, *newshape):
        """a.copyless_reshape(shape)

        Returns an array containing the same data with a new shape.
        Will raise an error if the data in `a` cannot be reshaped
        to the newshape without performing a copy (e.g. due
        to strides that are irregular and prevent dimensions
        from being collapsed together).

        Refer to `dali.copyless_reshape` for full documentation.

        See Also
        --------
        Array.reshape : equivalent function, will not raise error if copy required
        dali.reshape : equivalent function, will not raise error if copy required
        dali.copyless_reshape : equivalent function
        """
        cdef vector[int] cnewshape = list_from_args(newshape)
        return Array.wrapc(self.o.reshape(cnewshape))

    def debug_memory(Array self, bint print_contents=False):
        """a.debug_memory(print_contents=False)

        Returns a string containing low-level information about
        the state of the memory used for the array `a`, its
        location, and freshness across devices.
        """
        cdef stringstream ss
        cdef CSynchronizedMemory* mem = self.o.memory().get()
        mem[0].debug_info(ss, print_contents, self.o.dtype())
        return ss.to_string().decode("utf-8")

    def __str__(Array self):
        cdef stringstream ss
        self.o.print_me(ss)
        return "Array(" + ss.to_string().decode("utf-8") + ")"

    def __repr__(Array self):
        return str(self)

    def expand_dims(Array self, int axis):
        """a.expand_dims(axis)

        Expand the shape of an array.

        Insert a new axis, corresponding to a given position in the array shape.

        Parameters
        ----------
        axis : int
            Position (amongst axes) where new axis is to be inserted.

        Returns
        -------
        res : Array
            Output array. The number of dimensions is one greater than that of
            the input array.
        """
        return Array.wrapc(self.o.expand_dims(axis))

    def broadcast_axis(Array self, int axis):
        """a.broadcast_axis(axis)

        Make one of the axis of the array become broadcasted.

        Replace the dimension at axis with a broadcasted dimension.

        Parameters
        ----------
        axis : int
            Position (amongst axes) where a dimension should be made into a broadcasted dimension.

        Returns
        -------
        res : Array
            Output array. The number of dimensions is equal to that of the input array.
        """
        return Array.wrapc(self.o.broadcast_axis(axis))

    def insert_broadcast_axis(Array self, int new_axis):
        """a.insert_broadcast_axis(axis)

        Expand the shape of an array with a broadcasted dimension.

        Insert a new broadcast axis, corresponding to a given position in the array shape.

        Parameters
        ----------
        axis : int
            Position (amongst axes) where new broadcasted axis is to be inserted.

        Returns
        -------
        res : Array
            Output array. The number of dimensions is one greater than that of
            the input array.
        """
        return Array.wrapc(self.o.insert_broadcast_axis(new_axis))

    def broadcast_scalar_to_ndim(Array self, int ndim):
        """a.broadcast_scalar_to_ndim(ndim)

        Constructs an N-dimensional array that broadcasts (repeats)
        the scalar in every dimension.

        Parameters
        ----------
        ndim : int
            Desired new dimensionality for the scalar (ndim must be >= 0)

        Returns
        -------
        res : Array
            Output array. An array with `ndim` dimensions all equal to 1,
            that are all broadcasted.
        """
        return Array.wrapc(self.o.broadcast_scalar_to_ndim(ndim))

