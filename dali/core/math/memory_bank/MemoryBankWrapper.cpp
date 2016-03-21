#include "MemoryBankWrapper.h"
#include <dali/math/SynchronizedMemory.h>

template<typename R>
void MemoryBankWrapper<R>::clear_cpu() {
	SynchronizedMemory<R>::clear_cpu();
}
template<typename R>
void MemoryBankWrapper<R>::clear_gpu() {
	#ifdef DALI_USE_CUDA
		SynchronizedMemory<R>::clear_gpu();
	#else
		throw std::runtime_error("Dali when compiled without cuda cannot clear GPU.");
	#endif
}

template class MemoryBankWrapper<float>;
template class MemoryBankWrapper<double>;
template class MemoryBankWrapper<int>;
