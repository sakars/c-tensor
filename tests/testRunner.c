#ifdef TEST_BUILD
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>


extern void tensor_alloc_suite_builder(void);



void (*constructors[])(void) = {
	tensor_alloc_suite_builder,
	(void(*)(void))NULL /*Sentinel*/
};

int main(){
	CU_initialize_registry();
	void (**iterator)(void) = constructors;
	while(*iterator){
		(*iterator)();
		iterator++;
	}
	//CU_pSuite suite = CU_add_suite("main", NULL, NULL);
	//CU_add_test(suite, "first", test_data);
	CU_basic_run_tests();
	CU_cleanup_registry();
	return 0;
}

#endif