#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;
#define USE_CPU_TIMER 1
#define USE_GPU_TIMER 1

#define	IN
#define OUT
#define INOUT

#if USE_CPU_TIMER == 1
__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }
#else
#define CHECK_TIME_START
#define CHECK_TIME_END(a)
#endif

#if USE_GPU_TIMER == 1
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

void create_device_timer()
{
	CUDA_CALL(cudaEventCreate(&cuda_timer_start));
	CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_device_timer()
{
	CUDA_CALL(cudaEventDestroy(cuda_timer_start));
	CUDA_CALL(cudaEventDestroy(cuda_timer_stop));
}

inline void start_device_timer()
{
	cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
	cudaEventSynchronize(cuda_timer_stop);

	cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }
#else
#define CHECK_TIME_INIT_GPU()
#define CHECK_TIME_START_GPU()
#define CHECK_TIME_END_GPU(a)
#define CHECK_TIME_DEST_GPU()
#endif

#define N_SIZE (1 << 26)
#define NF_SIZE (1 << 6)

#define NO_SHARED 0
#define SHARED 1

#define BLOCK_SIZE (1 << 6)

#define BLOCK_WIDTH (1 << 3)
#define BLOCK_HEIGHT (BLOCK_SIZE / BLOCK_WIDTH)

#define N_ITERATION (1 << 0)

TIMER_T compute_time = 0;
TIMER_T device_time = 0;

int N;
int Nf;

extern __shared__ float sharedBuffer[ ];

int *h_ArrayElements;
int *h_SumOfArrayElements_CPU;
int *h_SumOfArrayElements_GPU_No_Shared;
int *h_SumOfArrayElements_GPU_Shared;

cudaError_t Sum_n_elements_GPU(IN int *p_ArrayElements, OUT int *p_SumOfElements_GPU, int Nf, int Shared_flag);

__global__ void Sum_n_elements_Kernel_No_shared(IN int *d_ArrayElements, OUT int *d_SumOfArrayElements, int N, int Nf) {
	const unsigned block_id = blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned id = block_id * BLOCK_SIZE + thread_id;

	int sum = 0;

	for (int i = -Nf; i <= Nf; i++) {
		if (id + i >= N || id + i < 0) continue;
		sum += d_ArrayElements[id + i];
	}

	d_SumOfArrayElements[id] = sum;
}

__global__ void Sum_n_elements_Kernel_shared(IN int *d_ArrayElements, OUT int *d_SumOfArrayElements, int N, int Nf) {
	const unsigned block_id	 = blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned id = block_id * BLOCK_SIZE + thread_id;

	int i;
	if(thread_id == 0){
		for(i = 0; i < Nf; i++){
			if(id+i <Nf) sharedBuffer[i] = 0;
			else sharedBuffer[i] = d_ArrayElements[id + i - Nf];
		}
	}

	if(thread_id == BLOCK_SIZE - 1){
		for(i = 1; i <= Nf; i++){
			if(id + i >= N) sharedBuffer[thread_id + Nf + i] = 0;
			else sharedBuffer[thread_id + Nf + i] =d_ArrayElements[id + i];
		}
	}

	__syncthreads();

	sharedBuffer[thread_id + Nf] = d_ArrayElements[id];

	int sum = 0;
	for(i = -Nf; i <= Nf; i++){
		sum += sharedBuffer[thread_id + i + Nf];
	}

	d_SumOfArrayElements[id] = sum;
}

void Sum_n_elements_CPU(IN int *p_ArrayElements, OUT int *p_SumOfElements_CPU, int Nf) {
	int i, j, sum;

	for (i = 0; i < N; i++) {
		sum = 0;
		for (j = -Nf; j <= Nf; j++) {
			if (i + j >= N || i + j < 0) continue;
			sum += p_ArrayElements[i + j];
		}
		p_SumOfElements_CPU[i] = sum;
	}
}

void read_bin_file() {
	printf("***Binary File Read Start!!\n");
	FILE *fp = fopen("gen.bin", "rb");
	fread(&N, sizeof(int), 1, fp);
	fread(&Nf, sizeof(int), 1, fp);

	h_ArrayElements = (int *)malloc(N * sizeof(int));
	h_SumOfArrayElements_CPU = (int *)malloc(N * sizeof(int));
	h_SumOfArrayElements_GPU_No_Shared = (int *)malloc(N * sizeof(int));
	h_SumOfArrayElements_GPU_Shared = (int *)malloc(N * sizeof(int));

	fread(h_ArrayElements, sizeof(int), N, fp);

	fclose(fp);
	printf("***Binary File Read End!!\n\n");
}

void init_bin_file(IN int n, IN int nf) {
	printf("***Binary File Create Start!!\n");
	srand((unsigned)time(NULL));
	FILE *fp = fopen("gen.bin", "wb");
	fwrite(&n, sizeof(int), 1, fp);
	fwrite(&nf, sizeof(int), 1, fp);

	int i, input;

	for (i = 0; i < n; i++) {
		input = (int)((float)rand() / RAND_MAX * 200 - 100);
		fwrite(&input, sizeof(int), 1, fp);
	}

	fclose(fp);
	printf("***Binary File Create End!!\n\n");
}

int main()
{
	int i;
	init_bin_file(N_SIZE, NF_SIZE);
	read_bin_file();

	TIMER_T CPU_time = 0.0f, GPU_time_NO_SHARED = 0.0f, GPU_time_SHARED;

	for (i = 0; i < N_ITERATION; i++) {
		CHECK_TIME_START;
		Sum_n_elements_CPU(h_ArrayElements, h_SumOfArrayElements_CPU, Nf);
		CHECK_TIME_END(compute_time);
		CPU_time += compute_time;

		Sum_n_elements_GPU(h_ArrayElements, h_SumOfArrayElements_GPU_No_Shared, Nf, NO_SHARED);
		GPU_time_NO_SHARED += device_time;

		Sum_n_elements_GPU(h_ArrayElements, h_SumOfArrayElements_GPU_Shared, Nf, SHARED);
		GPU_time_SHARED += device_time;
	}

	for (i = 0; i < N; i++) {
		if (h_SumOfArrayElements_CPU[i] != h_SumOfArrayElements_GPU_No_Shared[i] || h_SumOfArrayElements_CPU[i] != h_SumOfArrayElements_GPU_Shared[i]) {
			printf("%d : CPU : %d,\tGPU no shared : %d,\tGPU shared : %d\n", i, h_SumOfArrayElements_CPU[i], h_SumOfArrayElements_GPU_No_Shared[i], h_SumOfArrayElements_GPU_Shared[i]);
			break;
		}
	}
	if (i == N)
		printf("***Kernel execution Success!!\n\n");

	printf("***CPU compute time : %.3f ms\n", CPU_time / N_ITERATION);
	printf("***GPU NO SHARED compute time : %.3f ms\n", GPU_time_NO_SHARED / N_ITERATION);
	printf("***GPU SHARED compute time : %.3f ms\n", GPU_time_SHARED / N_ITERATION);

	free(h_ArrayElements);
	free(h_SumOfArrayElements_CPU);
	free(h_SumOfArrayElements_GPU_No_Shared);
	free(h_SumOfArrayElements_GPU_Shared);

	return 0;
}

cudaError_t Sum_n_elements_GPU(IN int *p_ArrayElements, OUT int *p_SumOfElements_GPU, int Nf, int Shared_flag) {
	cudaError_t cudaStatus;

	CUDA_CALL(cudaSetDevice(0));

	int *d_ArrayElements, *d_SumOfElements;
	size_t mem_size;

	mem_size = N * sizeof(int);
	CUDA_CALL(cudaMalloc(&d_ArrayElements, mem_size));
	CUDA_CALL(cudaMalloc(&d_SumOfElements, mem_size));

	CUDA_CALL(cudaMemcpy(d_ArrayElements, p_ArrayElements, mem_size, cudaMemcpyHostToDevice));

	dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 gridDim(N / BLOCK_SIZE); 
	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();

	switch (Shared_flag)
	{
	case NO_SHARED:
		Sum_n_elements_Kernel_No_shared << <gridDim, blockDim >> > (d_ArrayElements, d_SumOfElements, N, Nf);
		break;
	case SHARED:
		Sum_n_elements_Kernel_shared  << <gridDim, blockDim , sizeof(float) * (BLOCK_SIZE + 2 * Nf) >> > (d_ArrayElements, d_SumOfElements, N, Nf);
		break;
	}

	CUDA_CALL(cudaDeviceSynchronize());
	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	CUDA_CALL(cudaMemcpy(p_SumOfElements_GPU, d_SumOfElements, mem_size, cudaMemcpyDeviceToHost));
	
	cudaFree(d_ArrayElements);
	cudaFree(d_SumOfElements);

	return cudaStatus;
}
