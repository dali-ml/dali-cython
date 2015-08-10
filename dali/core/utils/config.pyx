cdef extern from "dali/math/SynchronizedMemory.h":
    cdef enum Device:
        DEVICE_GPU,
        DEVICE_CPU
    Device default_preferred_device

cdef class Config:
    property default_device:
        def __get__(self):
            global default_preferred_device
            if default_preferred_device == DEVICE_CPU:
                return 'cpu'
            elif default_preferred_device == DEVICE_GPU:
                return 'gpu'
            else:
                raise ValueError("default_preferred_device is not set correctly.")

        def __set__(self, device):
            assert(type(device) == str), "Device must be a string (gpu, cpu)."
            global default_preferred_device
            if device.lower() == 'cpu':
                default_preferred_device = DEVICE_CPU
            elif device.lower() == 'gpu':
                default_preferred_device = DEVICE_GPU
            else:
                raise ValueError("Device must be one of cpu, gpu.")

config = Config()

__all__ = [ 'config' ]
