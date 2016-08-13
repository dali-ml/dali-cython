from ..array.array cimport ensure_array
from ..array.op.unary cimport c_identity

import dali
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

cpdef Tensor ensure_tensor(object arr) except +:
    if type(arr) == Tensor:
        return arr
    else:
        return Tensor(Array(arr, borrow=True))

cdef vector[CTensor] ensure_tensor_list(object tensors) except +:
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

cdef list ctensors_to_list(const vector[CTensor]& ctensors) except +:
    out = []
    for i in range(ctensors.size()):
        out.append(Tensor.wrapc(ctensors[i]))
    return out

cdef class DoNotInitialize:
    """For internal use only - it exists so that Array initialization
    can be deferred to C++ functions."""
    pass

cdef class Tensor:
    def __cinit__(Tensor self, object data, dtype=None, preferred_device=None, borrow=False):
        if type(data) == DoNotInitialize:
            return
        cdef Array arr

        arr = Array(data, dtype, preferred_device, borrow)

        self.o = CTensor(arr.o, False)

    @staticmethod
    cdef Tensor wrapc(CTensor o):
        ret = Tensor(DoNotInitialize())
        ret.o = o
        return ret

    cdef c_np.NPY_TYPES cdtype(Tensor self):
        return dtype_dali_to_c_np(self.o.dtype())

    property dtype:
        def __get__(Tensor self):
            return dtype_dali_to_np(self.o.dtype())

    property constant:
        def __get__(Tensor self):
            return self.o.constant

        def __set__(Tensor self, bint constant):
            self.o.constant = constant

    property w:
        def __get__(Tensor self):
            return Array.wrapc(self.o.w)

        def __set__(Tensor self, other):
            self.o.w.operator_assign(c_identity((<Array>ensure_array(other)).o, True))

    property dw:
        def __get__(Tensor self):
            return Array.wrapc(self.o.dw)

        def __set__(Tensor self, other):
            self.o.dw.operator_assign(c_identity((<Array>ensure_array(other)).o, True))

    property shape:
        def __get__(Tensor self):
            return tuple(self.o.shape())

    property preferred_device:
        def __get__(Tensor self):
            return Device.wrapc(self.o.preferred_device())

    property number_of_elements:
        def __get__(Tensor self):
            return self.o.number_of_elements()

    property size:
        def __get__(Tensor self):
            return self.o.number_of_elements()

    property ndim:
        def __get__(Tensor self):
            return self.o.ndim()

    property T:
        def __get__(Tensor self):
            return self.transpose()

    def transpose(Tensor self, *axes):
        """t.transpose(*axes)

        Returns a view of the tensor with axes transposed.

        For a 1-D tensor, this has no effect.
        For a 2-D tensor, this is the usual matrix transpose.
        For an n-D tensor, if axes are given, their order indicates how the
        axes are permuted. If axes are not provided and
        ``t.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``t.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : omitted, tuple of ints, or `n` ints

         * no argument: reverses the order of the axes.

         * tuple of ints: `i` in the `j`-th place in the tuple means `t`'s
           `i`-th axis becomes `t.transpose()`'s `j`-th axis.

         * `n` ints: same as an n-tuple of the same ints (this form is
           intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : Tensor
            View of `t`, with axes suitably permuted.

        See Also
        --------
        Tensor.T : Tensor property returning the tensor transposed.
        """
        cdef vector[int] cdims
        if len(axes) == 0:
            return Tensor.wrapc(self.o.transpose())
        else:
            caxes = list_from_args(axes)
            return Tensor.wrapc(self.o.transpose(caxes))

    def swapaxes(Tensor self, int axis1, int axis2):
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

    def grad(Tensor self):
        """
        a.grad()

        Add the sum of the contents of this Tensor
        to the global objective function.
        You can obtain gradients for each parameter Tensor
        involved in the objective function by calling
        ``dali.backward()``.

        See Also
        --------
        dali.backward : perform backpropagation
        """
        self.o.grad()

    def clear_grad(Tensor self):
        """
        a.clear_grad()

        Erase the gradient storage for this Tensor.
        Fills the Array `a.dw` with zeros. Inplace
        operation.
        """

    def clear(Tensor self):
        """a.clear()

        Inplace operation that replaces all the contents of
        the Tensor with zeros.
        Warning: Do not call this operation if the contents
        of this Tensor were already needed for backpropagation,
        as the gradients computed relying on this Tensor
        will now be incorrect.
        """
        self.o.clear()

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

    def relu(Tensor self):
        """
        a.relu()

        Rectified linear nonlinearity.

        Refer to `dali.relu` for full documentation.

        See Also
        --------
        dali.relu : equivalent function
        """
        return Tensor.wrapc(self.o.relu())

    def sigmoid(Tensor self):
        """
        a.sigmoid()

        Sigmoid nonlinearity.

        Refer to `dali.sigmoid` for full documentation.

        See Also
        --------
        dali.sigmoid : equivalent function
        dali.steep_sigmoid : sigmoid nonlinearity with controllable slope
        Tensor.steep_sigmoid : sigmoid nonlinearity with controllable slope
        """
        return Tensor.wrapc(self.o.sigmoid())

    def steep_sigmoid(Tensor self, float aggressiveness=3.75):
        """
        a.steep_sigmoid(aggressiveness=3.75)

        Sigmoid nonlinearity with controllable slope.

        Refer to `dali.steep_sigmoid` for full documentation.

        See Also
        --------
        dali.steep_sigmoid : equivalent function
        dali.sigmoid : computes sigmoid with usual slope
        Tensor.sigmoid : computes sigmoid with usual slope
        """
        return Tensor.wrapc(self.o.steep_sigmoid(aggressiveness))

    def tanh(Tensor self):
        """
        a.tanh()

        Hyperbolic tangent nonlinearity.

        Refer to `dali.tanh` for full documentation.

        See Also
        --------
        dali.tanh : equivalent function
        """
        return Tensor.wrapc(self.o.tanh())

    def log(Tensor self):
        """
        a.log()

        Natural logarithm.

        Refer to `dali.log` for full documentation.

        See Also
        --------
        dali.log : equivalent function
        """
        return Tensor.wrapc(self.o.log())

    def exp(Tensor self):
        """
        a.exp()

        Exponential

        Refer to `dali.exp` for full documentation.

        See Also
        --------
        dali.exp : equivalent function
        """
        return Tensor.wrapc(self.o.exp())

    def abs(Tensor self):
        """
        a.abs()

        Absolute value.

        Refer to `dali.abs` for full documentation.

        See Also
        --------
        dali.abs : equivalent function
        """
        return Tensor.wrapc(self.o.abs())

    def softplus(Tensor self):
        """
        a.softplus()

        Soft plus nonlinearity.

        Refer to `dali.softplus` for full documentation.

        See Also
        --------
        dali.softplus : equivalent function
        """
        return Tensor.wrapc(self.o.softplus())

    def sqrt(Tensor self):
        """
        a.sqrt()

        Square root.

        Refer to `dali.sqrt` for full documentation.

        See Also
        --------
        dali.sqrt : equivalent function
        """
        return Tensor.wrapc(self.o.sqrt())

    def rsqrt(Tensor self):
        """
        a.rsqrt()

        Take the reciprocal of the square root.

        Refer to `dali.rsqrt` for full documentation.

        See Also
        --------
        dali.rsqrt : equivalent function
        """
        return Tensor.wrapc(self.o.rsqrt())

    def square(Tensor self):
        """
        a.square()

        Take the square.

        Refer to `dali.square` for full documentation.

        See Also
        --------
        dali.square : equivalent function
        """
        return Tensor.wrapc(self.o.square())

    def cube(Tensor self):
        """
        a.cube()

        Take the cube.

        Refer to `dali.cube` for full documentation.

        See Also
        --------
        dali.cube : equivalent function
        """
        return Tensor.wrapc(self.o.cube())

    def eltinv(Tensor self):
        """
        a.eltinv()

        Take the element-wise reciprocal.

        Refer to `dali.eltinv` for full documentation.

        See Also
        --------
        dali.eltinv : equivalent function
        dali.reciprocal : equivalent function
        Tensor.reciprocal : equivalent function
        """
        return Tensor.wrapc(self.o.eltinv())

    def reciprocal(Tensor self):
        """
        a.reciprocal()

        Take the element-wise reciprocal.

        Refer to `dali.reciprocal` for full documentation.

        See Also
        --------
        dali.eltinv : equivalent function
        dali.reciprocal : equivalent function
        Tensor.eltinv : equivalent function
        """
        return Tensor.wrapc(self.o.eltinv())

    def sum(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.sum())
        else:
            return Tensor.wrapc(self.o.sum(axis))

    def mean(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.mean())
        else:
            return Tensor.wrapc(self.o.mean(axis))

    def min(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.min())
        else:
            return Tensor.wrapc(self.o.min(axis))

    def max(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.max())
        else:
            return Tensor.wrapc(self.o.max(axis))

    def L2_norm(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.L2_norm())
        else:
            return Tensor.wrapc(self.o.L2_norm(axis))

    def argmin(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.argmin())
        else:
            return Tensor.wrapc(self.o.argmin(axis))

    def argmax(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.argmax())
        else:
            return Tensor.wrapc(self.o.argmax(axis))

    def argsort(Tensor self, axis=None):
        if axis is None:
            return Tensor.wrapc(self.o.argsort())
        else:
            return Tensor.wrapc(self.o.argsort(axis))

    def dot(Tensor self, other):
        """
        a.dot(b)

        Dot product of two tensors.

        Refer to `dali.dot` for full documentation.

        See Also
        --------
        dali.dot : equivalent function
        """
        return Tensor.wrapc(self.o.dot(ensure_tensor(other).o))

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
            step size is 1. If `step` is specified, `start` must also be given.
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
    def gaussian(double mean=0.0, double std=1.0, vector[int] shape=(), dtype=np.float32, preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.gaussian(mean, std, shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def uniform(double low=1.0, high=None, vector[int] shape=(), dtype=np.float32, preferred_device=None):
        """
        Creates a Tensor filled with numbers from distribution

            Uniform(low, high)

        if high is left out or None, then low is treated as both
        lower and upper bound, so that the numbers come from
        the distribtion

            Uniform(-low, low)

        For example if low = -1 and high = 1 then array may contain
        number -0.9, 0.0, 0.4, 0.99, but may not contain -1.5 or 10.

        Parameters
        ----------
        low: double
            lower bound for the uniform distribution
        high: double
            upper bound for the uniform distribution
        shape: [int]
            shape of the output Tensor
        dtype: dali.dtype
            dtype of the output Tensor
        preferred_device: dali.Device
            preferred device for data storage. If it is equal to None,
            a dali.default_device() is used.

        Returns
        -------
        out: Tensor
            tensor containing numbers from uniform distribution
        """
        if high is None:
            high = low
            low = -low

        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.uniform(low, <double>high, shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def bernoulli(double prob=0.5, vector[int] shape=(), dtype=np.float32, preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.bernoulli(prob, shape, dtype_np_to_dali(dtype), device.o))

    @staticmethod
    def bernoulli_normalized(double prob=0.5, vector[int] shape=(), dtype=np.float32, preferred_device=None):
        cdef Device device = ensure_device(preferred_device)
        return Tensor.wrapc(CTensor.bernoulli_normalized(prob, shape, dtype_np_to_dali(dtype), device.o))

    def __getitem__(Tensor self, args):
        if not isinstance(args, tuple):
            args = (args,)

        cdef int arg_as_int
        cdef CBroadcast br

        cdef bint                    use_tensor = True
        cdef CTensor                 ten       = self.o
        cdef CSlicingInProgressTensor slicing

        for arg in args:
            if type(arg) == slice:
                if use_tensor:
                    slicing   = ten.operator_bracket(parse_slice(arg))
                    use_tensor = False
                else:
                    slicing = slicing.operator_bracket(parse_slice(arg))
            elif type(arg) == int:
                arg_as_int = arg
                if use_tensor:
                    ten = ten.operator_bracket(arg_as_int)
                else:
                    slicing = slicing.operator_bracket(arg_as_int)
            elif arg is None:
                if use_tensor:
                    slicing = ten.operator_bracket(br)
                    use_tensor = False
                else:
                    slicing = slicing.operator_bracket(br)
            else:
                if use_tensor:
                    ten = ten.operator_bracket(ensure_tensor(arg).o)
                else:
                    ten = slicing.totensor()
                    ten = ten.operator_bracket(ensure_tensor(arg).o)
                    use_tensor = True
        if use_tensor:
            return Tensor.wrapc(ten)
        else:
            return Tensor.wrapc(slicing.totensor())

    def __int__(Tensor self):
        return <int>self.o.w

    def __float__(Tensor self):
        return <float>self.o.w

    def __add__(self, other):
        return dali.add(self, other)

    def __radd__(self, other):
        return dali.add(other, self)

    def __len__(Tensor self):
        cdef vector[int] shape
        if self.o.ndim() != 0:
            shape = self.o.shape()
            return shape[0]
        else:
            raise TypeError("len() of unsized object.")

    def __sub__(self, other):
        return dali.sub(self, other)

    def __rsub__(self, other):
        return dali.sub(other, self)

    def __mul__(self, other):
        return dali.eltmul(self, other)

    def __rmul__(self, other):
        return dali.eltmul(other, self)

    def __setstate__(Tensor self, state):
        self.o = CTensor((<Array>state["w"]).o, False)
        self.constant = state["cst"]

    def __getstate__(Tensor self):
        state = {
            "w": self.w, "cst": self.constant
        }
        return state

    def __reduce__(Tensor self):
        return (
            self.__class__,
            (
                DoNotInitialize(),
            ), self.__getstate__(),
        )
