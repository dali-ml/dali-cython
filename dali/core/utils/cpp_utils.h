#ifndef UTILS_CPP_UTILS_H
#define UTILS_CPP_UTILS_H

#include <string>
#include <dali/utils/gpu_utils.h>

void print_str(std::string s);

void set_default_gpu(int);
std::string get_gpu_name(int device);
int num_gpus();

#endif

