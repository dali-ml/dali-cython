cdef extern from "dali/math/SynchronizedMemory.h":
    cdef enum Device:
        DEVICE_GPU,
        DEVICE_CPU
    Device default_preferred_device

cdef extern from "dali/utils/cpp_utils.h" nogil:
    void set_default_device_to_gpu() except +
    void set_default_device_to_cpu()

cdef class Config:
    property default_device:
        def __get__(self):
            if default_preferred_device == DEVICE_CPU:
                return 'cpu'
            elif default_preferred_device == DEVICE_GPU:
                return 'gpu'
            else:
                raise ValueError("default_preferred_device is not set correctly.")

        def __set__(self, device):
            assert(type(device) == str), "Device must be a string (gpu, cpu)."
            if device.lower() == 'cpu':
                set_default_device_to_cpu()
            elif device.lower() == 'gpu':
                set_default_device_to_gpu()
            else:
                raise ValueError("Device must be one of cpu, gpu.")

config = Config()

__all__ = [ 'config' ]
