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
        DeviceT type() const
        int number()   const
        string description(bint real_gpu_name) const
        bint is_fake() const

        @staticmethod
        CDevice fake(int number)

        bint is_cpu() const

        @staticmethod
        CDevice cpu()

        @staticmethod
        CDevice device_of_doom()

        @staticmethod
        vector[CDevice] installed_devices()

        bint is_gpu() const

        @staticmethod
        CDevice gpu(int number)

        @staticmethod
        int num_gpus()
