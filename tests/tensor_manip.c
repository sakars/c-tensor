#include "tensor.h"
#include <CUnit/CUnit.h>

static void test_tensor_swapaxis(void) {
	TensorObject tensor = Tensor_new(2, (TensorShape_t[]){2, 3});
	TensorData_t* x = Tensor_get(&tensor, (TensorShape_t[]){1, 2});
	Tensor_swapaxis(&tensor, 0, 1);
	TensorData_t* y = Tensor_get(&tensor, (TensorShape_t[]){2, 1});
	CU_ASSERT_EQUAL(x, y);
	Tensor_free(&tensor);
}

// Test if tensor slice is a view of the original tensor from different axis
static void test_tensor_slice(void) {
	TensorObject tensor = Tensor_new(2, (TensorShape_t[]){2, 3});
	TensorData_t* x = Tensor_get(&tensor, (TensorShape_t[]){1, 2});
	TensorObject slice = Tensor_slice(&tensor, 0, 1);
	TensorData_t* y = Tensor_get(&slice, (TensorShape_t[]){2});
	CU_ASSERT_EQUAL(tensor.data, slice.data);
	CU_ASSERT_EQUAL(tensor.ref_count, slice.ref_count);
	CU_ASSERT_EQUAL(tensor.offset + tensor.strides[0], slice.offset);
	TensorObject slice2 = Tensor_slice(&tensor, 1, 2);
	TensorData_t* z = Tensor_get(&slice2, (TensorShape_t[]){1});
	CU_ASSERT_EQUAL(tensor.data, slice2.data);
	CU_ASSERT_EQUAL(tensor.ref_count, slice2.ref_count);
	CU_ASSERT_EQUAL(tensor.offset + 2 * tensor.strides[1], slice2.offset);
	CU_ASSERT_EQUAL(x, y);
	CU_ASSERT_EQUAL(x, z);
	Tensor_free(&tensor);
	Tensor_free(&slice);
	Tensor_free(&slice2);
}

static void test_tensor_slice_of_slice(void) {
	TensorObject tensor = Tensor_new(2, (TensorShape_t[]){2, 3});
	TensorData_t* x = Tensor_get(&tensor, (TensorShape_t[]){1, 1});
	TensorObject slice = Tensor_slice(&tensor, 0, 1);
	TensorObject slice2 = Tensor_slice(&slice, 0, 1);
	TensorData_t* y = Tensor_get(&slice2, (TensorShape_t[]){1});
	CU_ASSERT_EQUAL(tensor.data, slice2.data);
	CU_ASSERT_EQUAL(tensor.ref_count, slice2.ref_count);
	CU_ASSERT_EQUAL(x, y);
	Tensor_free(&tensor);
	Tensor_free(&slice);
	Tensor_free(&slice2);
}

void tensor_manip_suite_builder(void) {
	CU_pSuite suite = CU_add_suite("tensor_manipulation", NULL, NULL);
	CU_add_test(suite, "tensor_swapaxis", test_tensor_swapaxis);
	CU_add_test(suite, "tensor_slice", test_tensor_slice);
	CU_add_test(suite, "tensor_slice_of_slice", test_tensor_slice_of_slice);
}