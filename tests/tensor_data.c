#include "tensor.h"
#include <CUnit/CUnit.h>

static void test_tensor_pointer_access(void) {
	TensorObject tensor = Tensor_new(3, (TensorShape_t[]){2, 18, 5});
	for(TensorShape_t i = 0; i < 2; i++) {
		for(TensorShape_t j = 0; j < 18; j++) {
			for(TensorShape_t k = 0; k < 5; k++) {
				TensorData_t* x = Tensor_get(&tensor, (TensorShape_t[]){i, j, k});
				CU_ASSERT_EQUAL(x, tensor.data + tensor.offset + k + 5 * j + 5 * 18 * i);
			}
		}
	}
	Tensor_free(&tensor);
}

static void test_tensor_set(void) {
	TensorObject tensor = Tensor_new(3, (TensorShape_t[]){2, 18, 5});
	for(TensorShape_t i = 0; i < 2; i++) {
		for(TensorShape_t j = 0; j < 18; j++) {
			for(TensorShape_t k = 0; k < 5; k++) {
				Tensor_set(&tensor, (TensorShape_t[]){i, j, k}, k + 5 * j + 5 * 18 * i);
			}
		}
	}
	for(TensorShape_t i = 0; i < 2; i++) {
		for(TensorShape_t j = 0; j < 18; j++) {
			for(TensorShape_t k = 0; k < 5; k++) {
				TensorData_t* x = Tensor_get(&tensor, (TensorShape_t[]){i, j, k});
				CU_ASSERT_EQUAL(*x, k + 5 * j + 5 * 18 * i);
			}
		}
	}
	Tensor_free(&tensor);
}

static void test_tensor_write_to(void) {
	TensorObject tensor = Tensor_new(3, (TensorShape_t[]){2, 3, 5});
	TensorObject tensor2 = Tensor_new(3, (TensorShape_t[]){2, 3, 5});
	for(TensorShape_t i = 0; i < 2; i++) {
		for(TensorShape_t j = 0; j < 3; j++) {
			for(TensorShape_t k = 0; k < 5; k++) {
				Tensor_set(&tensor, (TensorShape_t[]){i, j, k}, k + 5 * j + 5 * 3 * i);
			}
		}
	}
	Tensor_write_to(&tensor, &tensor2);
	for(TensorShape_t i = 0; i < 2; i++) {
		for(TensorShape_t j = 0; j < 3; j++) {
			for(TensorShape_t k = 0; k < 5; k++) {
				TensorData_t* x = Tensor_get(&tensor2, (TensorShape_t[]){i, j, k});
				CU_ASSERT_EQUAL(*x, k + 5 * j + 5 * 3 * i);
			}
		}
	}
	Tensor_free(&tensor);
	Tensor_free(&tensor2);
}

void tensor_data_suite_builder(void) {
	CU_pSuite suite = CU_add_suite("tensor_data_access", NULL, NULL);
	CU_add_test(suite, "tensor_pointer_access", test_tensor_pointer_access);
	CU_add_test(suite, "tensor_set", test_tensor_set);
	CU_add_test(suite, "tensor_write_to", test_tensor_write_to);
}