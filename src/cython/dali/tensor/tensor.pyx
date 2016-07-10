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
