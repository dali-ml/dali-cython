from dali.array.memory.device cimport CDevice

cdef extern from "dali/runtime_config.h":
    cdef CDevice default_preferred_device "memory::default_preferred_device"
    cdef bint use_cudnn
