#include "tensor.h"
#include <CUnit/CUnit.h>

// Test 2 dimensional tensor allocation
static void test_tensor_alloc() {
	TensorObject tensor = Tensor_new(2, (TensorShape_t[]){2, 3});
	CU_ASSERT_EQUAL(tensor.ndim, 2);
	CU_ASSERT_EQUAL(tensor.shape[0], 2);
	CU_ASSERT_EQUAL(tensor.shape[1], 3);
	CU_ASSERT_EQUAL(tensor.strides[0], 3);
	CU_ASSERT_EQUAL(tensor.strides[1], 1);
	Tensor_free(&tensor);
	CU_ASSERT_PTR_NULL(tensor.shape);
	CU_ASSERT_PTR_NULL(tensor.strides);
	CU_ASSERT_PTR_NULL(tensor.data);
}

// Test 3 dimensional tensor allocation
static void test_tensor_alloc_2() {
	TensorObject tensor = Tensor_new(3, (TensorShape_t[]){2, 3, 4});
	CU_ASSERT_EQUAL(tensor.ndim, 3);
	CU_ASSERT_EQUAL(tensor.shape[0], 2);
	CU_ASSERT_EQUAL(tensor.shape[1], 3);
	CU_ASSERT_EQUAL(tensor.shape[2], 4);
	CU_ASSERT_EQUAL(tensor.strides[0], 12);
	CU_ASSERT_EQUAL(tensor.strides[1], 4);
	CU_ASSERT_EQUAL(tensor.strides[2], 1);
	Tensor_free(&tensor);
	CU_ASSERT_PTR_NULL(tensor.shape);
	CU_ASSERT_PTR_NULL(tensor.strides);
	CU_ASSERT_PTR_NULL(tensor.data);
}

// Test that the allocations are the same after cloning
static void test_tensor_clone_allocations() {
	TensorObject tensor = Tensor_new(2, (TensorShape_t[]){2, 3});
	TensorObject clone = Tensor_clone(tensor);
	CU_ASSERT_EQUAL(clone.ndim, 2);
	CU_ASSERT_EQUAL(clone.shape[0], 2);
	CU_ASSERT_EQUAL(clone.shape[1], 3);
	CU_ASSERT_EQUAL(clone.strides[0], 3);
	CU_ASSERT_EQUAL(clone.strides[1], 1);
	Tensor_free(&tensor);
	CU_ASSERT_PTR_NULL(tensor.shape);
	CU_ASSERT_PTR_NULL(tensor.strides);
	CU_ASSERT_PTR_NULL(tensor.data);
	Tensor_free(&clone);
	CU_ASSERT_PTR_NULL(clone.shape);
	CU_ASSERT_PTR_NULL(clone.strides);
	CU_ASSERT_PTR_NULL(clone.data);
}

// Test that the data is the same after cloning
static void test_tensor_clone_data() {
	TensorObject tensor = Tensor_new(2, (TensorShape_t[]){2, 3});
	Tensor_set(&tensor, (TensorShape_t[]){0, 0}, 1.0);
	Tensor_set(&tensor, (TensorShape_t[]){0, 1}, 2.0);
	Tensor_set(&tensor, (TensorShape_t[]){0, 2}, 3.0);
	Tensor_set(&tensor, (TensorShape_t[]){1, 0}, 4.0);
	Tensor_set(&tensor, (TensorShape_t[]){1, 1}, 5.0);
	Tensor_set(&tensor, (TensorShape_t[]){1, 2}, 6.0);
	TensorObject clone = Tensor_clone(tensor);
	// test that the data is the same
	CU_ASSERT_EQUAL(*Tensor_get(&clone, (TensorShape_t[]){0, 0}), 1.0);
	CU_ASSERT_EQUAL(*Tensor_get(&clone, (TensorShape_t[]){0, 1}), 2.0);
	CU_ASSERT_EQUAL(*Tensor_get(&clone, (TensorShape_t[]){0, 2}), 3.0);
	CU_ASSERT_EQUAL(*Tensor_get(&clone, (TensorShape_t[]){1, 0}), 4.0);
	CU_ASSERT_EQUAL(*Tensor_get(&clone, (TensorShape_t[]){1, 1}), 5.0);
	CU_ASSERT_EQUAL(*Tensor_get(&clone, (TensorShape_t[]){1, 2}), 6.0);
	Tensor_free(&tensor);
	Tensor_free(&clone);
	
}

void tensor_alloc_suite_builder(void) {
	CU_pSuite suite = CU_add_suite("tensor_alloc", NULL, NULL);
	CU_add_test(suite, "tensor_alloc_2d", test_tensor_alloc);
	CU_add_test(suite, "tensor_alloc_3d", test_tensor_alloc_2);
	CU_add_test(suite, "tensor_clone", test_tensor_clone_allocations);
	CU_add_test(suite, "tensor_clone_data", test_tensor_clone_data);
}