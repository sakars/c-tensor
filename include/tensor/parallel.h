
#include "tensor.h"
#pragma once
#define TENSOR_LOOP_NUM_THREADS 4
struct loop_args {
	TensorObject obj;
	TensorShape_t idx;
	void *args;
};

void Tensor_loop_over_dim(TensorObject obj, TensorShape_t axis, void* (*func)(void *args), void *args);
void Tensor_init(void);
void Tensor_cleanup(void);