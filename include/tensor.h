
#pragma once
typedef double TensorData_t;
typedef unsigned int TensorShape_t;

struct _TensorObject {
	TensorShape_t ndim;
	TensorShape_t *shape;
	TensorShape_t *strides;
	TensorData_t *data;
	TensorShape_t *ref_count;
	TensorShape_t offset;
};

/// @brief Tensor object type. It is not recommended to access the members directly. Not thread safe.
/// @details The members of this struct are not guaranteed to be stable.
/// 	   They are definitely not thread safe.
/// 	   For multi-threaded applications, use the functions in tensor.h
typedef struct _TensorObject TensorObject;

TensorObject Tensor_new(const TensorShape_t ndim, const TensorShape_t *const shape_template);
void Tensor_free(TensorObject *tensor);
TensorObject Tensor_clone(TensorObject tensor);
int Tensor_swapaxis(TensorObject *tensor, const TensorShape_t axis1, const TensorShape_t axis2);
TensorObject Tensor_slice(TensorObject *tensor, const TensorShape_t axis, const TensorShape_t idx);
TensorData_t* Tensor_get(TensorObject *tensor, const TensorShape_t *const idx);
void Tensor_set(TensorObject *tensor, const TensorShape_t *const idx, const TensorData_t value);
int Tensor_write_to(TensorObject *src_tensor, TensorObject *dst_tensor);


