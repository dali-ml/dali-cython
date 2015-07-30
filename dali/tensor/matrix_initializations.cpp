#include "matrix_initializations.h"
#include "dali/tensor/Weights.h"

template<typename R>
Mat<R> matrix_initializations<R>::uniform(R low, R high, int rows, int cols) {
    return Mat<R>(rows, cols, weights<R>::uniform(low, high));
}
template<typename R>
Mat<R> matrix_initializations<R>::gaussian(R mean, R std, int rows, int cols) {
    return Mat<R>(rows, cols, weights<R>::gaussian(mean, std));
}
template<typename R>
Mat<R> matrix_initializations<R>::bernoulli(R prob, int rows, int cols) {
    return Mat<R>(rows, cols, weights<R>::bernoulli(prob));
}

template<typename R>
Mat<R> matrix_initializations<R>::bernoulli_normalized(R prob, int rows, int cols) {
    return Mat<R>(rows, cols, weights<R>::bernoulli_normalized(prob));
}

template<typename R>
Mat<R> matrix_initializations<R>::eye(R diag, int width) {
    return Mat<R>(width, width, weights<R>::eye(diag));
}

template<typename R>
Mat<R> matrix_initializations<R>::empty(int rows, int cols) {
    return Mat<R>(rows, cols, weights<R>::empty());
}

template<typename R>
Mat<R> matrix_initializations<R>::ones(int rows, int cols) {
    return Mat<R>(rows, cols, weights<R>::ones());
}

template<typename R>
Mat<R> matrix_initializations<R>::from_pointer(R* ptr, int rows, int cols) {
	Mat<R> mat(rows, cols);
	if ((rows * cols) > 0) {
		// not actually allocated memory
		std::cout << "hello" << std::endl;
		mat.w().memory().cpu_ptr = ptr;
		mat.w().memory().allocated_cpu = true;
		mat.w().memory().cpu_fresh = true;
		std::cout << "hello" << std::endl;
		mat.w().memory().total_memory = rows * cols;
	}
	return mat;
}

template struct matrix_initializations<float>;
template struct matrix_initializations<double>;
template struct matrix_initializations<int>;
