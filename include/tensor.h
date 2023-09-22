

/** @mainpage
 * @section intro Introduction
 * @see TensorObject
 * @see TensorShape_t
*/


#pragma once

typedef double TensorData_t;


/**
 * @typedef TensorShape_t
 * @brief Tensor shape type.
 * @details This is the type used for the shape of a tensor.
 * 	   It is unsigned to prevent negative values.
 * 	   It is typedef'd to allow for easy changing of the type.
 * 	   It is used for the shape, strides, and reference count.
 * @see TensorObject
*/
typedef unsigned int TensorShape_t;

/** 
 * @struct TensorObject
 * @brief Tensor object.
 * 
 * 
 * @details This struct represents a tensor object. It contains the shape, strides, data, and reference count.
 * 	   The tensor object is reference counted. When a tensor object is created, the reference count is set to 1.
 * 	   When a tensor object is cloned, the reference count is incremented by 1.
 * 	   When a tensor object is freed, the reference count is decremented by 1.
 * 	   When a tensor object is sliced, the reference count is incremented by 1.
 * 	   Clones and slices share the same data and reference counter.
 * 	   Multi-threaded slicing is not supported.
 * 
 * @see Tensor_new
 * @see Tensor_free
 * @see Tensor_clone
 * @see Tensor_slice
 * @see Tensor_get
 * @see Tensor_set
 * @see Tensor_write_to
 * @see Tensor_swapaxis
 * @see TensorShape_t
 * @see TensorData_t
 * @see Tensor_zeros
 * @see Tensor_zeros_like
 * @see Tensor_ones
 * @see Tensor_ones_like
 * @see Tensor_eye
 * @see Tensor_eye_like
 * @see Tensor_random_uniform
 * @see Tensor_random_normal
*/
struct TensorObject {
	TensorShape_t ndim; ///< Number of dimensions.
	TensorShape_t *shape; ///< Array of length ndim containing the size of each dimension.
	TensorShape_t *strides; ///< Array of length ndim containing the stride of each dimension.
	TensorData_t *data; ///< Array of length total_size containing the data.
	TensorShape_t *ref_count; ///< Reference count pointer.
	TensorShape_t offset; ///< Offset of the first element.
		///< This is used when slicing a tensor to retain same data field for freeing.
};


typedef struct TensorObject TensorObject;

/**
 * @brief Create a new tensor.
 * @details This function creates a new tensor object.
 * 	   The shape of the tensor is specified by the ndim and shape_template parameters.
 * 	   The ndim parameter specifies the number of dimensions.
 * 	   The shape_template parameter is an array of length ndim containing the size of each dimension.
*/
TensorObject Tensor_new(const TensorShape_t ndim, const TensorShape_t *const shape_template);

/**
 * @brief Free a tensor.
 * @details This function frees a tensor object.
 * 	   If the reference count is 1, the tensor object is freed.
 * 	   If the reference count is greater than 1, the reference count is decremented by 1.
*/
void Tensor_free(TensorObject *tensor);

/**
 * @brief Clone a tensor.
 * @details This function clones a tensor object.
 *     The data isn't shared between the two tensors, only copied over.
*/
TensorObject Tensor_clone(TensorObject tensor);

/**
 * @brief Swap two axes of a tensor.
 * @details This function swaps two axes of a tensor object.
 *    The operation is done in-place.
 *   The axis1 and axis2 parameters specify the axes to swap.
 *  The axis1 and axis2 parameters must be less than the number of dimensions.
*/
int Tensor_swapaxis(TensorObject *tensor, const TensorShape_t axis1, const TensorShape_t axis2);

/**
 * @brief Slice a tensor along an axis.
 * @details This function slices a tensor object.
 * 	   The axis parameter specifies the axis to slice.
 * 	   The idx parameter specifies the index to slice at.
 * 	   The idx parameter must be less than the size of the axis.
 * 	   A new tensor object is created sharing the original data.
*/
TensorObject Tensor_slice(TensorObject *tensor, const TensorShape_t axis, const TensorShape_t idx);

/**
 * @brief Get a value from a tensor.
 * @details This function returns a pointer to the value from a tensor object.
 * 	   The idx parameter specifies the index to get.
 * 	   The idx parameter must be less than the size of each dimension.
 * 	   The returned pointer is valid until the tensor object is freed.
*/
TensorData_t* Tensor_get(TensorObject *tensor, const TensorShape_t *const idx);

/**
 * @brief Set a value in a tensor.
 * @details This function sets a value in a tensor object.
 * 	   The idx parameter specifies the index to set.
 * 	   The idx parameter must be less than the size of each dimension.
 * 	   The value parameter specifies the value to set.
 * 	   The value parameter is copied into the tensor object.
*/
void Tensor_set(TensorObject *tensor, const TensorShape_t *const idx, const TensorData_t value);

/**
 * @brief Write a tensor to another tensor.
 * @details This function writes a tensor object to another tensor object.
 *    This is similar to Tensor_clone, but the destination tensor object is used instead of creating a new one.
 *    The destination tensor object must have the same shape as the source tensor object.
 *    Useful when copying into slices. 
*/
int Tensor_write_to(TensorObject *src_tensor, TensorObject *dst_tensor);


