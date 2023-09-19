#ifndef TEST_BUILD
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "tensor/parallel.h"

int main() {
	Tensor_init();
	TensorObject to = Tensor_new(3, (TensorShape_t[]){6,2,23});
	for(TensorShape_t i=0;i<to.shape[0];i++){
		for(TensorShape_t j=0;j<to.shape[1];j++){
			for(TensorShape_t k=0;k<to.shape[2];k++){
				*Tensor_get(&to, (TensorShape_t[]){i, j, k}) = i + j + k;
			}
		}
	}
	Tensor_cleanup();
}

#endif

