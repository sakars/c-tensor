
#include "tensor.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "tensor/parallel.h"
// Loopies

static int killThreads=0;

static struct Job {
	void* (*func)(void *args);
	void* args;
	pthread_mutex_t* jobFinishedMutex;
	pthread_cond_t* jobFinished;
} Job;
static struct JobQueueElem {
	struct Job job;
	struct JobQueueElem* next;
} JobQueueElem;
static struct JobQueue {
	struct JobQueueElem* next;
	struct JobQueueElem* last;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
} jobQueue = {
	.next = NULL,
	.last = NULL
};

static void JobQueue_push(struct Job job) {
	struct JobQueueElem* elem = malloc(sizeof(struct JobQueueElem));
	elem->job = job;
	elem->next = NULL;
	if(jobQueue.next == NULL) {
		jobQueue.next = elem;
		jobQueue.last = elem;
	} else {
		jobQueue.last->next = elem;
		jobQueue.last = elem;
	}
	pthread_cond_signal(&jobQueue.cond);
}

static struct Job JobQueue_pop(void) {
	if(jobQueue.next == NULL) {
		pthread_mutex_unlock(&jobQueue.mutex);
		return (struct Job){NULL, NULL, NULL};
	}
	struct JobQueueElem* elem = jobQueue.next;
	jobQueue.next = elem->next;
	if(jobQueue.next == NULL) {
		jobQueue.last = NULL;
	}
	struct Job job = elem->job;
	free(elem);
	return job;
}

static struct WorkerThread {
	pthread_t thread;
	size_t id;
} *available_threads = NULL;

static const struct timespec THREAD_SLEEP = {
	.tv_sec = 10,
	.tv_nsec = 100000000
};

static void* WorkerThread_main(void* args) {
	struct WorkerThread* at = (struct WorkerThread*)args;
	while(1) {
		#ifdef DEBUG
			printf("Thread %lu waiting for queue lock\n", at->id);
		#endif
		
		pthread_mutex_lock(&jobQueue.mutex);
		
		#ifdef DEBUG
			printf("Thread %lu acquired queue lock\n", at->id);
		#endif
		
		while(jobQueue.next == NULL) {
			
			#ifdef DEBUG
				printf("Thread %lu queue empty\n", at->id);
				printf("Thread %lu releasing queue lock and waiting\n", at->id);
			#endif
			
			pthread_cond_timedwait(&jobQueue.cond, &jobQueue.mutex, &THREAD_SLEEP);
			if(killThreads) {
				pthread_mutex_unlock(&jobQueue.mutex);
				#ifdef DEBUG
					printf("Thread %lu exiting\n", at->id);
				#endif
				return NULL;
			}
		}
		struct Job job = JobQueue_pop();
		
		#ifdef DEBUG
			printf("Thread %lu got a job\n", at->id);
		#endif
		
		pthread_mutex_unlock(&jobQueue.mutex);
		
		#ifdef DEBUG
			printf("Thread %lu released queue lock, executing job.\n", at->id);
		#endif

		if(job.func == NULL) {
			// should not happen, but just in case.
			continue;
		}
		job.func(job.args);
		pthread_mutex_lock(job.jobFinishedMutex);
		pthread_cond_broadcast(job.jobFinished);
		pthread_mutex_unlock(job.jobFinishedMutex);

		#ifdef DEBUG
			printf("Thread %lu broadcasted job finished\n", at->id);
		#endif
		continue;
	}
	#ifdef DEBUG
		printf("Thread %lu exiting\n", at->id);
	#endif
	return NULL;
}

void Tensor_init(void) {
	available_threads = calloc(TENSOR_LOOP_NUM_THREADS, sizeof(struct WorkerThread));
	for(int i = 0; i < TENSOR_LOOP_NUM_THREADS; i++) {
		available_threads[i].id = i;
		pthread_create(&available_threads[i].thread, NULL, WorkerThread_main, &available_threads[i]);
		
	}
}


void Tensor_cleanup(void) {
	killThreads = 1;
	for(int i = 0; i < TENSOR_LOOP_NUM_THREADS; i++) {
		pthread_join(available_threads[i].thread, NULL);
	}
	free(available_threads);
	available_threads = NULL;
}

static void Tensor_addTask(void* (*func)(void *args), void* args, pthread_mutex_t* jobFinishedMutex, pthread_cond_t* jobFinished) {
	// if available_threads is NULL, then we need to initialize it.
	if(available_threads == NULL) {
		Tensor_init();
	}
	// construct the job and push it to the queue.
	pthread_mutex_lock(&jobQueue.mutex);

	#ifdef DEBUG
		printf("addTask acquired queue lock\n");
	#endif
	JobQueue_push((struct Job){
		.func = func, 
		.args = args, 
		.jobFinished = jobFinished,
		.jobFinishedMutex = jobFinishedMutex});
	pthread_mutex_unlock(&jobQueue.mutex);
	#ifdef DEBUG
		printf("addTask released queue lock\n");
	#endif
}


void Tensor_loop_over_dim(TensorObject obj, const TensorShape_t axis, void *(*func)(void *args), void* args) {
	// This should work, since no slice shares data with a different slice and by declaring The tensor non-thread safe,
	// we can guarantee that no other thread will access the tensor while this function is running.

	// allocate loop_args, threads and job finish flags.
	struct loop_args* loop_args = calloc(obj.shape[axis], sizeof(struct loop_args));
	pthread_cond_t* jobsFinished = calloc(obj.shape[axis], sizeof(pthread_cond_t));
	pthread_mutex_t* jobsFinishedMutex = calloc(obj.shape[axis], sizeof(pthread_mutex_t));
	{
		// alloc
		for(TensorShape_t i = 0; i < obj.shape[axis]; i++) {
			// generate the slices, fill in the loop_args and create threads.
			TensorObject slice = Tensor_slice(&obj, axis, i);
			loop_args[i].obj = slice;
			loop_args[i].idx = i;
			loop_args[i].args = args;
			//pthread_create(&threads[i], NULL, func, &loop_args[i]);
			pthread_mutex_init(&jobsFinishedMutex[i], NULL);
			pthread_cond_init(&jobsFinished[i], NULL);
			pthread_mutex_lock(&jobsFinishedMutex[i]);
			Tensor_addTask( func, &loop_args[i], &jobsFinishedMutex[i], &jobsFinished[i]);
		}
		// join
		for(TensorShape_t i = 0; i < obj.shape[axis]; i++) {
			// wait for the threads to complete all jobs and free the slices right after.
			pthread_cond_wait(&jobsFinished[i], &jobsFinishedMutex[i]);
			Tensor_free(&loop_args[i].obj);
		}
	}
	// free loop_args and threads.
	free(loop_args);
	free(jobsFinished);
	free(jobsFinishedMutex);
}
