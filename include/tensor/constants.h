/**
 * @file constants.h
 * @brief Constants header file.
 * @details This file contains the tensor object and function declarations centered around constants.
 * 
 * @see Tensor_zeros
 * @see Tensor_ones
 * @see Tensor_eye
 * @see Tensor_zeros_like
 * @see Tensor_ones_like
 * @see Tensor_eye_like
 * @see Tensor_random_uniform
 * @see Tensor_random_normal
 * 
*/
#include "tensor.h"

/**
 * @brief Create a new tensor object with all elements set to 0.
 * @param ndim Number of dimensions.
 * @param shape Array of length ndim containing the size of each dimension.
 * @return A new tensor object.
*/
TensorObject Tensor_zeros(TensorShape_t ndim, const TensorIndex_t *shape);

/**
 * @brief Create a new tensor object with all elements set to 1.
 * @param ndim Number of dimensions.
 * @param shape Array of length ndim containing the size of each dimension.
 * @return A new tensor object.
*/
TensorObject Tensor_ones(TensorShape_t ndim, const TensorIndex_t *shape);

/**
 * @brief Create a new tensor object with all elements set to the identity matrix.
 * @param size Size of the identity matrix.
 * @param col_count Number of columns in the identity matrix.
 * @param batch_size Number of identity matrices to create.
 * @return A new tensor object.
 * @details The identity matrix is a square matrix with 1s on the diagonal and 0s everywhere else.
 * 	   col_count is the number of columns in the identity matrix.
 * 	   It is required that size <= col_count.
 * 	   The identity matrix is batched by repeating it batch_size times.
*/
TensorObject Tensor_eye(TensorShape_t size, TensorShape_t col_count, TensorShape_t batch_size);

/**
 * @brief Create a new tensor object with all elements set to 0.
 * @param a Tensor object to copy the shape from.
 * @return A new tensor object.
*/
TensorObject Tensor_zeros_like(TensorObject a);

/**
 * @brief Create a new tensor object with all elements set to 1.
 * @param a Tensor object to copy the shape from.
 * @return A new tensor object.
*/
TensorObject Tensor_ones_like(TensorObject a);

/**
 * @brief Create a new tensor object with all elements set to the identity matrix.
 * @param a Tensor object to copy the shape from.
 * @return A new tensor object.
*/
TensorObject Tensor_eye_like(TensorObject a);

/**
 * @brief Create a new tensor object with all elements set to a random uniform value between min and max.
 * @param ndim Number of dimensions.
 * @param shape Array of length ndim containing the size of each dimension.
*/
TensorObject Tensor_random_uniform(TensorShape_t ndim, const TensorShape_t *shape, float min, float max);

/**
 * @brief Create a new tensor object with all elements set to a random normal value with mean and stddev.
 * @param ndim Number of dimensions.
 * @param shape Array of length ndim containing the size of each dimension.
 * @param mean Mean of the normal distribution.
 * @param stddev Standard deviation of the normal distribution.
 * @return A new tensor object.
*/
TensorObject Tensor_random_normal(TensorShape_t ndim, const TensorShape_t *shape, float mean, float stddev);

/**
 * @brief Create a new tensor object with all elements set to a random uniform value between min and max.
 * @param a Tensor object to copy the shape from.
 * @param min Minimum value of the uniform distribution.
 * @param max Maximum value of the uniform distribution.
 * @return A new tensor object.
 * @details The new tensor object will have the same shape as a.
 * 	   The new tensor object will have the same data type as a.
 * 	   The new tensor object will have all elements set to a random uniform value between min and max.
*/
TensorObject Tensor_random_uniform_like(TensorObject a, float min, float max);

/**
 * @brief Create a new tensor object with all elements set to a random uniform value between min and max.
 * @param shape Shape of the new tensor.
 * @param min Minimum value of the uniform distribution.
 * @param max Maximum value of the uniform distribution.
 * @return A new tensor object.
 * @details The new tensor object will have the same data type as the template parameter.
 * 	   The new tensor object will have all elements set to a random uniform value between min and max.
*/
TensorObject Tensor_random_normal_like(TensorObject a, float mean, float stddev);

/**
 * @brief Generate a random normal value with mean and stddev.
 * @param mean Mean of the normal distribution.
 * @param stddev Standard deviation of the normal distribution.
 * @return A random normal value.
*/
float generate_random_normal(float mean, float stddev);

/**
 * @brief Generate a random uniform value between min and max.
 * @param min Minimum value of the uniform distribution.
 * @param max Maximum value of the uniform distribution.
 * @return A random uniform value.
*/
float generate_random_uniform(float min, float max);
