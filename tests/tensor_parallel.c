#include "tensor.h"
#include <CUnit/CUnit.h>
#include "tensor/parallel.h"
#include <pthread.h>

struct ThreadCounter {
	pthread_mutex_t mutex;
	int count;
};

static void* _loop_over_dim_thread_count(void* args) {
	struct loop_args* loop_args = (struct loop_args*)args;
	struct ThreadCounter* thc = (struct ThreadCounter*)loop_args->args;
	pthread_mutex_lock(&thc->mutex);
	thc->count++;
	pthread_mutex_unlock(&thc->mutex);

	return NULL;
}

static void test_loop_over_thread_count(void) {
	TensorObject to = Tensor_new(3, (TensorShape_t[]){6,2,23});
	struct ThreadCounter thc;
	thc.count=0;
	pthread_mutex_init(&thc.mutex, NULL);

	thc.count=0;
	Tensor_loop_over_dim(to, 0, _loop_over_dim_thread_count, (void*)&thc);
	CU_ASSERT_EQUAL(thc.count, 6);

	thc.count=0;
	Tensor_loop_over_dim(to, 1, _loop_over_dim_thread_count, (void*)&thc);
	CU_ASSERT_EQUAL(thc.count, 2);

	thc.count=0;
	Tensor_loop_over_dim(to, 2, _loop_over_dim_thread_count, (void*)&thc);
	CU_ASSERT_EQUAL(thc.count, 23);

	pthread_mutex_destroy(&thc.mutex);
	Tensor_free(&to);
}

static void* _loop_over_dim_write(void* args) {
	struct loop_args* loop_args = (struct loop_args*)args;
	TensorObject to = loop_args->obj;
	for(TensorShape_t i=0;i<to.shape[0];i++){
		*Tensor_get(&to, (TensorShape_t[]){i}) = i;
	}
	return NULL;
}

static void test_loop_over_write(void) {
	TensorObject to = Tensor_new(2, (TensorShape_t[]){2, 6});
	Tensor_loop_over_dim(to, 1, _loop_over_dim_write, (void*)&to);
	for(TensorShape_t j=0;j<to.shape[1];j++){
		for(TensorShape_t i=0;i<to.shape[0];i++){
			CU_ASSERT_EQUAL(*Tensor_get(&to, (TensorShape_t[]){i, j}), i);
		}
	}
	Tensor_free(&to);
}

void tensor_parallel_suite_builder(void) {
	CU_pSuite suite = CU_add_suite("tensor_manipulation", NULL, NULL);
	CU_add_test(suite, "loop_over_dim_thread_count", test_loop_over_thread_count);
	CU_add_test(suite, "loop_over_dim_write", test_loop_over_write);

}