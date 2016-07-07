from third_party.libcpp11.stringstream cimport stringstream

from .device          cimport CDevice
from dali.array.dtype cimport *

cdef extern from "dali/array/memory/synchronized_memory.h" namespace "memory":
    cdef cppclass CSynchronizedMemory "memory::SynchronizedMemory":
        CDevice preferred_device
        int total_memory
        int inner_dimension

        bint is_fresh(const CDevice& device)

        bint is_allocated(const CDevice& CDevice)

        bint is_any_fresh()

        bint is_any_allocated()

        void lazy_clear()
        void clear()

        bint allocate(const CDevice& CDevice)
        void free(const CDevice& CDevice)

        void move_to(const CDevice& CDevice)
        void to_gpu(const int& gpu_number)
        void to_cpu()
        void* data(const CDevice& CDevice)
        void* readonly_data(const CDevice& CDevice)
        void* mutable_data(const CDevice& CDevice)
        void* overwrite_data(const CDevice& device)

        void debug_info(stringstream& stream, bint print_contents, DType dtype)
