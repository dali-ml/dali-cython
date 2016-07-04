#ifndef DALI_CORE_MATH_MEMORY_BANK_MEMORY_BANK_WRAPPER_H
#define DALI_CORE_MATH_MEMORY_BANK_MEMORY_BANK_WRAPPER_H
template<typename R>
class MemoryBankWrapper {
	public:
		static void clear_gpu();
		static void clear_cpu();
};

#endif
