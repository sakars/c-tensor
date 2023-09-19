
#include "tensor.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "tensor/parallel.h"
// Loopies

static int killThreads=0;

static struct AvailableThread {
	pthread_mutex_t mutex;
	pthread_t thread;
	int available;
	void* (*func)(void *args);
	void* args;
	int* jobFinished;
} *available_threads = NULL;

static void* Thread_handler(void* args) {
	struct AvailableThread* at = (struct AvailableThread*)args;
	while (1)
	{
		if(killThreads) {
			return NULL;
		}
		pthread_mutex_lock(&at->mutex);
		if(at->available) {
			pthread_mutex_unlock(&at->mutex);
			continue;
		}
		at->func(at->args);
		at->func = NULL;
		at->args = NULL;
		at->available = 1;
		*at->jobFinished = 1;
		pthread_mutex_unlock(&at->mutex);
	}
	return NULL;
}

void Tensor_init(void) {
	available_threads = calloc(TENSOR_LOOP_NUM_THREADS, sizeof(struct AvailableThread));
	for(int i = 0; i < TENSOR_LOOP_NUM_THREADS; i++) {
		pthread_mutex_init(&available_threads[i].mutex, NULL);
		available_threads[i].available = 1;
		pthread_create(&available_threads[i].thread, NULL, Thread_handler, &available_threads[i]);
		// lock the mutex to prevent the thread from running.
		// pthread_mutex_lock(&available_threads[i].mutex);
	}
}

void Tensor_cleanup(void) {
	killThreads = 1;
	for(int i = 0; i < TENSOR_LOOP_NUM_THREADS; i++) {
		
		pthread_join(available_threads[i].thread, NULL);
		pthread_mutex_lock(&available_threads[i].mutex);
		pthread_mutex_destroy(&available_threads[i].mutex);
	}
	free(available_threads);
	available_threads = NULL;
}

static struct AvailableThread* Tensor_addTask(void* (*func)(void *args), void* args, int* jobFinished) {
	// find an available thread.
	while(1) {
		for(int i = 0; i < TENSOR_LOOP_NUM_THREADS; i++) {
			int response = pthread_mutex_trylock(&available_threads[i].mutex);
			if(response != 0) {
				continue;
			}
			if(available_threads[i].available) {
				available_threads[i].func = func;
				available_threads[i].args = args;
				available_threads[i].available = 0;
				available_threads[i].jobFinished = jobFinished;
				pthread_mutex_unlock(&available_threads[i].mutex);
				return &available_threads[i];
			}
			pthread_mutex_unlock(&available_threads[i].mutex);
		}
	}
}


void Tensor_loop_over_dim(TensorObject obj, const TensorShape_t axis, void *(*func)(void *args), void* args) {
	// This should work, since no slice shares data with a different slice and by declaring The tensor non-thread safe,
	// we can guarantee that no other thread will access the tensor while this function is running.

	// allocate loop_args and threads.
	struct loop_args* loop_args = calloc(obj.shape[axis], sizeof(struct loop_args));
	int* jobsFinished = calloc(obj.shape[axis], sizeof(int));
	struct AvailableThread** threads = calloc(obj.shape[axis], sizeof(struct AvailableThread*));
	{
		// alloc
		for(TensorShape_t i = 0; i < obj.shape[axis]; i++) {
			// generate the slices, fill in the loop_args and create threads.
			TensorObject slice = Tensor_slice(&obj, axis, i);
			loop_args[i].obj = slice;
			loop_args[i].idx = i;
			loop_args[i].args = args;
			//pthread_create(&threads[i], NULL, func, &loop_args[i]);
			threads[i] = Tensor_addTask( func, &loop_args[i], &jobsFinished[i]);
		}
		// join
		for(TensorShape_t i = 0; i < obj.shape[axis]; i++) {
			// join threads and free slices.
			//pthread_join(threads[i], NULL);
			while(1) {
				// wait for the thread to finish.
				pthread_mutex_lock(&threads[i]->mutex);
				if(jobsFinished[i]) {
					pthread_mutex_unlock(&threads[i]->mutex);
					break;
				}
				pthread_mutex_unlock(&threads[i]->mutex);
			}
			Tensor_free(&loop_args[i].obj);
		}
	}
	// free loop_args and threads.
	free(loop_args);
	free(jobsFinished);
	free(threads);
}
