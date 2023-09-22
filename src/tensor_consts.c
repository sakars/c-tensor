#include "tensor/constants.h"
#include <stdlib.h>
#include <math.h>

float generate_random_uniform(float min, float max) {
	int r = rand();
	float f = (float)r / RAND_MAX;
	return min + f * (max - min);
}

static int generate_random_normal_initialized = 0;

static const int GEN_RAN_NOR_MIX_U = 0;
static const int GEN_RAN_NOR_MIX_V = 1;
static int generate_random_normal_genval = GEN_RAN_NOR_MIX_U;
float generate_random_normal(float mean, float stddev) {
	// Box-Muller transform with polar form
	static float u, v, s, mul;
	if(!generate_random_normal_initialized) u = generate_random_uniform(-1, 1);
	
	if(generate_random_normal_genval == GEN_RAN_NOR_MIX_U) while(1) {
		v = generate_random_uniform(-1, 1);
		s = u * u + v * v;
		if(s==0 || s >= 1) continue;
		break;
	}
	mul = sqrtf(-2 * logf(s) / s);
	if(generate_random_normal_genval == GEN_RAN_NOR_MIX_U) {
		generate_random_normal_genval = GEN_RAN_NOR_MIX_V;
		return mean + stddev * u * mul;
	} else {
		generate_random_normal_genval = GEN_RAN_NOR_MIX_U;
		return mean + stddev * v * mul;
	}
}

TensorObject Tensor_zeros(TensorShape_t ndim, const TensorIndex_t *shape) {
	TensorObject t = Tensor_create(ndim, shape);
	TensorShape_t total_size = 1;
	for(TensorShape_t i=0; i<ndim; i++) total_size *= shape[i];
	for(TensorShape_t i=0; i<total_size; i++) t->data[i] = 0;
	return t;
}

TensorObject Tensor_zeros_like(TensorObject a) {
	return Tensor_zeros(a->ndim, a->shape);
}

TensorObject Tensor_ones(TensorShape_t ndim, const TensorIndex_t *shape) {
	TensorObject t = Tensor_create(ndim, shape);
	TensorShape_t total_size = 1;
	for(TensorShape_t i=0; i<ndim; i++) total_size *= shape[i];
	for(TensorShape_t i=0; i<total_size; i++) t->data[i] = 1;
	return t;
}

TensorObject Tensor_ones_like(TensorObject a) {
	return Tensor_ones(a->ndim, a->shape);
}

TensorObject Tensor_eye(TensorShape_t size, TensorShape_t col_count, TensorShape_t batch_size) {
	TensorShape_t shape[3] = {batch_size, size, col_count};
	TensorObject t = Tensor_zeros(3, shape);
	for(TensorShape_t i=0; i<batch_size; i++) {
		for(TensorShape_t j=0; j<size; j++) t->data[i*size*col_count + j*col_count + j] = 1;
	}
	return t;
}

TensorObject Tensor_eye_like(TensorObject a) {
	if(a->ndim == 1) return Tensor_eye(a->shape[0], a->shape[0], 1);
	if(a->ndim == 2) return Tensor_eye(a->shape[0], a->shape[1], 1);
	return Tensor_eye(a->shape[1], a->shape[2], a->shape[0]);
}