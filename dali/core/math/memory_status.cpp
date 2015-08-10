#include "memory_status.h"


template<typename R>
bool is_gpu_fresh(const Mat<R>& mat) {
    #ifdef DALI_USE_CUDA
        if (!mat.empty()) {
            return mat.w().memory().gpu_fresh;
        } else {
            return false;
        }
    #else
        return false;
    #endif
}

template<typename R>
bool is_cpu_fresh(const Mat<R>& mat) {
    if (!mat.empty()) {
        return mat.w().memory().cpu_fresh;
    } else {
        return false;
    }
}

template<typename R>
bool is_gpu_allocated(const Mat<R>& mat) {
    #ifdef DALI_USE_CUDA
        if (!mat.empty()) {
            return mat.w().memory().allocated_gpu;
        } else {
            return false;
        }
    #else
        return false;
    #endif
}

template<typename R>
bool is_cpu_allocated(const Mat<R>& mat) {
    if (!mat.empty()) {
        return mat.w().memory().allocated_cpu;
    } else {
        return false;
    }
}


template bool is_gpu_fresh(const Mat<int>& mat);
template bool is_gpu_fresh(const Mat<float>& mat);
template bool is_gpu_fresh(const Mat<double>& mat);

template bool is_cpu_fresh(const Mat<int>& mat);
template bool is_cpu_fresh(const Mat<float>& mat);
template bool is_cpu_fresh(const Mat<double>& mat);

template bool is_gpu_allocated(const Mat<int>& mat);
template bool is_gpu_allocated(const Mat<float>& mat);
template bool is_gpu_allocated(const Mat<double>& mat);

template bool is_cpu_allocated(const Mat<int>& mat);
template bool is_cpu_allocated(const Mat<float>& mat);
template bool is_cpu_allocated(const Mat<double>& mat);
