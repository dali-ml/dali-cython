#ifndef DALI_MATH_MEMORY_STATUS_H
#define DALI_MATH_MEMORY_STATUS_H


#include "dali/tensor/Mat.h"

template<typename R>
bool is_gpu_fresh(const Mat<R>& mat);

template<typename R>
bool is_cpu_fresh(const Mat<R>& mat);

template<typename R>
bool is_gpu_allocated(const Mat<R>& mat);

template<typename R>
bool is_cpu_allocated(const Mat<R>& mat);

template<typename R>
void to_cpu(const Mat<R>& mat);

template<typename R>
void to_gpu(const Mat<R>& mat);

#endif
