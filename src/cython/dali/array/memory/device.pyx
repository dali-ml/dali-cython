from .device import *

cdef class Device:
    cdef CDevice o

    def __cinit__(Device self):
        self.o = CDevice.cpu()

