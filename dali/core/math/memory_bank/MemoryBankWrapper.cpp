#include "MemoryBankWrapper.h"
#include <dali/math/memory_bank/MemoryBank.h>

template<typename R>
void MemoryBankWrapper<R>::clear_cpu() {
	memory_bank<R>::clear_cpu();
}
template<typename R>
void MemoryBankWrapper<R>::clear_gpu() {
	#ifdef DALI_USE_CUDA
		memory_bank<R>::clear_gpu();
	#else
		throw std::runtime_error("Dali when compiled without cuda cannot clear GPU.");
	#endif
}

template class MemoryBankWrapper<float>;
template class MemoryBankWrapper<double>;
template class MemoryBankWrapper<int>;
