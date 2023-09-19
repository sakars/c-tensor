#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"

// ---Allocations---

/**
 * @brief      Allocates a new tensor object.
 * @details    This function allocates a new tensor object. The tensor object
 * 		   is initialized with the given shape template. The shape template
 * 		   is an array of TensorShape_t, which is an unsigned integer type.
 * 		   The shape template is copied into the tensor object, so the
 * 		   shape template can be freed after the tensor object is created.
 * 		   The tensor object requires freeing when it is no longer needed.
 * @see 	  Tensor_free
*/
TensorObject Tensor_new(const TensorShape_t ndim, const TensorShape_t *const shape_template) {
	TensorObject tensor;
	tensor.ndim = ndim;
	TensorShape_t total_size = 1;
	for(TensorShape_t i = 0; i < ndim; i++) {
		total_size *= shape_template[i];
	}
	tensor.shape = (TensorShape_t*)calloc(tensor.ndim, sizeof(TensorShape_t));
	for(TensorShape_t i = 0; i < tensor.ndim; i++) {
		tensor.shape[i] = shape_template[i];
	}
	tensor.strides = (TensorShape_t*)calloc(tensor.ndim, sizeof(TensorShape_t));
	tensor.strides[tensor.ndim - 1] = 1;
	for(int i = tensor.ndim - 2; i >= 0; i--) {
		tensor.strides[i] = tensor.strides[i + 1] * tensor.shape[i + 1];
	}
	tensor.data = (TensorData_t*)calloc(total_size, sizeof(TensorData_t));
	tensor.offset = 0;
	tensor.ref_count = (TensorShape_t*)malloc(sizeof(TensorShape_t));
	*tensor.ref_count = 1;
	return tensor;
}

void Tensor_free(TensorObject *tensor) {
	free(tensor->shape);
	tensor->shape = NULL;
	free(tensor->strides);
	tensor->strides = NULL;
	if(*tensor->ref_count > 1) {
		*tensor->ref_count -= 1;
		return;
	}
	free(tensor->data);
	tensor->data = NULL;
	free(tensor->ref_count);
	tensor->ref_count = NULL;
}

/**
 * @brief      Clones a tensor.
 * @details    This function clones a tensor. The new tensor object contains the same data,
 * 		   but is not tied to the original tensor object. The new tensor object
 * 		   requires freeing when it is no longer needed.
 * @see 	  Tensor_free
*/
TensorObject Tensor_clone(TensorObject tensor) {
	TensorObject clone;
	clone.ndim = tensor.ndim;
	clone.shape = (TensorShape_t*)calloc(clone.ndim, sizeof(TensorShape_t));
	clone.strides = (TensorShape_t*)calloc(clone.ndim, sizeof(TensorShape_t));
	clone.offset = 0;
	clone.ref_count = (TensorShape_t*)malloc(sizeof(TensorShape_t));
	*clone.ref_count = 1;
	// copy shape and calculate total size
	TensorShape_t total_size = 1;
	for(TensorShape_t i = 0; i < clone.ndim; i++) {
		clone.shape[i] = tensor.shape[i];
		total_size *= clone.shape[i];
	}
	// allocate data
	clone.data = (TensorData_t*)calloc(total_size, sizeof(TensorData_t));
	// generate strides
	clone.strides[clone.ndim - 1] = 1;
	for(int i = clone.ndim - 2; i >= 0; i--) {
		clone.strides[i] = clone.strides[i + 1] * clone.shape[i + 1];
	}
	// copy data taking into account offset and strides
	TensorData_t *original_iter = tensor.data + tensor.offset;
	TensorData_t *clone_iter = clone.data;
	TensorShape_t *idx = (TensorShape_t*)calloc(clone.ndim, sizeof(TensorShape_t));
	for(TensorShape_t i = 0; i < total_size; i++) {
		*clone_iter = *original_iter;
		clone_iter++;
		original_iter+= tensor.strides[tensor.ndim - 1];
		idx[clone.ndim - 1]++;
		for(TensorShape_t j = clone.ndim - 1; j > 0; j--) {
			if(idx[j] == clone.shape[j]) {
				idx[j] = 0;
				original_iter -= tensor.strides[j] * tensor.shape[j];
				original_iter += tensor.strides[j - 1];
				idx[j - 1]++;
			}
		}
	}
	free(idx);
	return clone;
}

// ---Manipulations---

/**
 * @brief      Swaps two axes of a tensor.
 * @details    This function swaps two axes of a tensor. The tensor object
 * 		   is modified in place.
 * @param      tensor  The tensor object
 * @param      axis1   The first axis
 * @param      axis2   The second axis
*/
int Tensor_swapaxis(TensorObject *tensor, const TensorShape_t axis1, const TensorShape_t axis2) {
	if(axis1 >= tensor->ndim || axis2 >= tensor->ndim) {
		return -1;
	}
	TensorShape_t temp = tensor->shape[axis1];
	tensor->shape[axis1] = tensor->shape[axis2];
	tensor->shape[axis2] = temp;
	temp = tensor->strides[axis1];
	tensor->strides[axis1] = tensor->strides[axis2];
	tensor->strides[axis2] = temp;
	return 0;
}

/**
 * @brief      Slices a tensor along a given axis.
 * @details    This function returns a new tensor object that is a slice of the
 * 		   original tensor along the given axis. The new tensor object
 * 		   shares the same data as the original tensor object, thus requires
 * 		   freeing.
*/
TensorObject Tensor_slice(TensorObject *tensor, const TensorShape_t axis, const TensorShape_t idx) {
	TensorObject slice;
	slice.ndim = tensor->ndim - 1;
	slice.shape = (TensorShape_t*)calloc(slice.ndim, sizeof(TensorShape_t));
	slice.strides = (TensorShape_t*)calloc(slice.ndim, sizeof(TensorShape_t));
	slice.data = tensor->data;
	slice.offset = tensor->offset + idx * tensor->strides[axis];
	slice.ref_count = tensor->ref_count;
	(*slice.ref_count)++;
	for(TensorShape_t i = 0; i < axis; i++) {
		slice.shape[i] = tensor->shape[i];
		slice.strides[i] = tensor->strides[i];
	}
	for(TensorShape_t i = axis; i < slice.ndim; i++) {
		slice.shape[i] = tensor->shape[i + 1];
		slice.strides[i] = tensor->strides[i + 1];
	}
	return slice;
}

// ---Data access---

TensorData_t* Tensor_get(TensorObject *tensor, const TensorShape_t *const idx) {
	TensorData_t *data = tensor->data + tensor->offset;
	for(TensorShape_t i = 0; i < tensor->ndim; i++) {
		data += idx[i] * tensor->strides[i];
	}
	return data;
}

void Tensor_set(TensorObject *tensor, const TensorShape_t *const idx, const TensorData_t value) {
	TensorData_t *data = tensor->data + tensor->offset;
	for(TensorShape_t i = 0; i < tensor->ndim; i++) {
		data += idx[i] * tensor->strides[i];
	}
	*data = value;
}

/**
 * @brief      Writes a tensor to another tensor.
 * @details    This function writes a tensor to another tensor. The two tensors
 * 		   must have the same shape. The two tensors can be the same
 * 		   tensor object.
*/
int Tensor_write_to(TensorObject *src_tensor, TensorObject *dst_tensor) {
	if(src_tensor->ndim != dst_tensor->ndim) {
		return -1;
	}
	for(TensorShape_t i = 0; i < src_tensor->ndim; i++) {
		if(src_tensor->shape[i] != dst_tensor->shape[i]) {
			return -1;
		}
	}
	TensorData_t *src_data = src_tensor->data + src_tensor->offset;
	TensorData_t *dst_data = dst_tensor->data + dst_tensor->offset;
	TensorShape_t total_size = 1;
	for(TensorShape_t i = 0; i < src_tensor->ndim; i++) {
		total_size *= src_tensor->shape[i];
	}
	for(TensorShape_t i = 0; i < total_size; i++) {
		dst_data[i] = src_data[i];
	}
	return 0;
}

