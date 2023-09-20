
#include "tensor.h"
#pragma once
#define TENSOR_LOOP_NUM_THREADS 4

struct loop_args {
	TensorObject obj;
	TensorShape_t idx;
	void *args;
};

/**
 * @brief Loop over a dimension of a tensor.
 * @details This function will loop over a dimension of a tensor.
 * The function will call the function func with the arguments args wrapped inside loop_args.
 * The function func should return NULL.
 * The function func is allowed to modify only the given slice of the tensor.
 * 
*/
void Tensor_loop_over_dim(TensorObject obj, TensorShape_t axis, void* (*func)(void *args), void *args);


/**
 * @brief Initialize the thread pool.
 * @details This function should be called before any thread pool functions are called.
*/
void Tensor_init(void);

/**
 * @brief Cleanup the thread pool.
 * @details This function should be called after all thread pool functions are called.
 * This function will wait for all threads to finish their current job before exiting.
 * This function will also free all memory allocated by the thread pool.
 * Although this function is not strictly necessary as the OS will free all memory allocated by the thread pool,
 * it is recommended to call this function.
*/
void Tensor_cleanup(void);