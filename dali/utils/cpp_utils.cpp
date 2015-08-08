#include "cpp_utils.h"

#include <iostream>
#include <dali/math/SynchronizedMemory.h>

void print_str(std::string s )  {
    std::cout << s << std::endl;
}

void set_default_device_to_gpu() {
    #ifdef DALI_USE_CUDA
        default_preferred_device = DEVICE_GPU;
    #else
        throw std::runtime_error("Dali compiled without GPU support. Cannot set GPU as preferred device.");
    #endif
}

void set_default_device_to_cpu() {
    default_preferred_device = DEVICE_CPU;
}
