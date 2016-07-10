import numpy as np

from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
c_np.import_array()

cdef object list_from_args(args):
    if len(args) > 0:
        if all(isinstance(arg, int) for arg in args):
            return args
        if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
            return args[0]
    raise ValueError("expected a list of integers")

cdef class Array:
    def __cinit__(Array self, vector[int] shape, dtype=np.float32, preferred_device=None):
        cdef Device device
        device = ensure_device(preferred_device)

        self.o = CArray(shape, dtype_np_to_dali(dtype), device.o)

    @staticmethod
    cdef Array wrapc(CArray o):
        ret = Array([])
        ret.o = o
        return ret

    cdef c_np.NPY_TYPES cdtype(Array self):
        return dtype_dali_to_c_np(self.o.dtype())

    property offset:
        def __get__(Array self):
            return self.o.offset()

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

    property number_of_elements:
        def __get__(Array self):
            return self.o.number_of_elements()

    property size:
        def __get__(Array self):
            return self.o.number_of_elements()

    def subshape(Array self):
        """
a.subshape()

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

    property T:
        def __get__(Array self):
            return self.transpose()

    def transpose(Array self, *axes):
        """
a.transpose(*axes)

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
        cdef vector[int] caxes
        if len(axes) == 0:
            return Array.wrapc(self.o.transpose())
        else:
            caxes = list_from_args(axes)
            return Array.wrapc(self.o.transpose(caxes))

    def squeeze(Array self, axis=None):
        """
a.squeeze(axis=None)

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
        """
a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `dali.swapaxes` for full documentation.

See Also
--------
dali.swapaxes : equivalent function
"""
        return Array.wrapc(self.o.swapaxes(axis1, axis2))

    def get_value(self, copy=False):
        """
a.get_value(copy=False)

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
                ndarray.strides[i] = strides[i] * dtype_to_itemsize(self.o.dtype())
        finally:
            free(np_shape)
        ndarray.base = <PyObject*> self
        Py_INCREF(self)
        return ndarray

    def ravel(Array self):
        """
a.ravel()

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
        """
a.copyless_ravel()

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
        """
a.reshape(shape)

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
        """
a.copyless_reshape(shape)

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
        """
a.debug_memory(print_contents=False)

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
        """
a.expand_dims(axis)

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
        """
a.broadcast_axis(axis)

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
        """
a.insert_broadcast_axis(axis)

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
        """
a.broadcast_scalar_to_ndim(ndim)

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

