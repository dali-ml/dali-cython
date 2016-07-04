#ifndef DALI_TENSOR_MATRIX_INITIALIZATIONS_H
#define DALI_TENSOR_MATRIX_INITIALIZATIONS_H

#include "dali/tensor/Mat.h"

template<typename R>
struct matrix_initializations {
	static Mat<R>* uniform(R low, R high, int rows, int cols);
	static Mat<R>* gaussian(R mean, R std, int rows, int cols);
	static Mat<R>* bernoulli(R prob, int rows, int cols);
	static Mat<R>* bernoulli_normalized(R prob, int rows, int cols);
	static Mat<R>* eye(R diag, int width);
	static Mat<R>* empty(int rows, int cols);
    static Mat<R>* ones(int rows, int cols);
    static Mat<R>* zeros(int rows, int cols);
	static Mat<R>* from_pointer(R* ptr, int rows, int cols);
    static Mat<R>* as_pointer(const Mat<R>& matrix);
};

#endif

