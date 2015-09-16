cdef extern from "dali/math/SynchronizedMemory.h":
    cdef enum Device:
        DEVICE_GPU,
        DEVICE_CPU
    Device default_preferred_device

cdef extern from "core/math/memory_bank/MemoryBankWrapper.h":
    cdef cppclass MemoryBankWrapper [T]:
        @staticmethod
        void clear_cpu()

        @staticmethod
        void clear_gpu() except +

cdef extern from "core/utils/cpp_utils.h" nogil:
    void set_default_gpu(int) except +
    string get_gpu_name(int device) except +
    int num_gpus() except +

cdef class Config:
    property num_gpus:
        def __get__(self):
            return num_gpus()

    property default_gpu:
        def __set__(self, int device):
            set_default_gpu(device)

    def gpu_id_to_name(self, int device_id):
        return get_gpu_name(device_id)

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

    def clear_gpu(self, dtype=np.float32):
        if dtype == np.float32:
            MemoryBankWrapper["float"].clear_gpu()
        elif dtype == np.float64:
            MemoryBankWrapper["double"].clear_gpu()
        elif dtype == np.int32:
            MemoryBankWrapper["int"].clear_gpu()
        else:
            raise ValueError("dtype must be one of np.float32, np.float64, or np.int32")

    def clear_cpu(self, dtype=np.float32):
        if dtype == np.float32:
            MemoryBankWrapper["float"].clear_cpu()
        elif dtype == np.float64:
            MemoryBankWrapper["double"].clear_cpu()
        elif dtype == np.int32:
            MemoryBankWrapper["int"].clear_cpu()
        else:
            raise ValueError("dtype must be one of np.float32, np.float64, or np.int32")


config = Config()

__all__ = [ 'config' ]
