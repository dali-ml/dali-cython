#include "cpp_utils.h"

#include <iostream>
#include <dali/math/SynchronizedMemory.h>

void print_str(std::string s )  {
    std::cout << s << std::endl;
}

void set_default_gpu(int gpu_id) {
    #ifdef DALI_USE_CUDA
        gpu_utils::set_default_gpu(gpu_id);
    #else
        throw std::runtime_error("Dali when compiled without cuda cannot set default GPU.");
    #endif
}

std::string get_gpu_name(int device) {
    #ifdef DALI_USE_CUDA
        return gpu_utils::get_gpu_name(device);
    #else
        throw std::runtime_error("Dali when compiled without cuda cannot get GPU name.");
        return "";
    #endif
}

int num_gpus() {
    #ifdef DALI_USE_CUDA
        return gpu_utils::num_gpus();
    #else
        throw std::runtime_error("Dali when compiled without cuda cannot get the number of GPUs.");
        return 0;
    #endif
}
