from libcpp.string cimport string
from third_party.libcpp11.vector cimport vector

cdef extern from "dali/array/memory/device.h" namespace "memory":
    enum DeviceT:
        DEVICE_T_ERROR = 0
        DEVICE_T_FAKE  = 1
        DEVICE_T_CPU   = 2
        DEVICE_T_GPU   = 3

    cdef cppclass CDevice "memory::Device":
        CDevice()
        DeviceT type()
        int number()
        string description(bint real_gpu_name)
        string gpu_name() except +
        bint is_fake()

        @staticmethod
        CDevice fake(int number)

        bint is_cpu()

        @staticmethod
        CDevice cpu()

        @staticmethod
        CDevice device_of_doom()

        @staticmethod
        vector[CDevice] installed_devices()

        bint is_gpu()

        @staticmethod
        CDevice gpu(int number)

        @staticmethod
        int num_gpus()


cpdef Device default_device()
cpdef void set_default_device(dev)
cpdef Device ensure_device(object dev)

cdef class Device:
    cdef CDevice o

    @staticmethod
    cdef Device wrapc(CDevice)
