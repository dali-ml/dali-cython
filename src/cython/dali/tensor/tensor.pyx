import numpy as np
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
c_np.import_array()

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
