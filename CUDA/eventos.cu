#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_20_atomic_functions.h"

#include <iostream>
using namespace std;

__device__ int dSum = 0;

__global__ void sum(int* d)
{
	int tid = threadIdx.x;
	//dSum += d[tid];

	atomicAdd(&dSum, d[tid]);
}

int main()
{
	const int count = 256;
	const int size = count * sizeof(int);

	int h[count];
	for (int i = 0; i < count; ++i)
		h[i] = i + 1;

	int* d;
	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	// Define 2 eventos CUDA
	cudaEvent_t start, end;

	// Cria os eventos
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Registra o primeiro evento
	cudaEventRecord(start);

	sum << <1, count >> >(d);

	// Registra o segundo evento
	cudaEventRecord(end);

	// Sincroniza o evento
	cudaEventSynchronize(end);

	// Calcula o tempo usado no processamento
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, end);

	int hSum;
	cudaMemcpyFromSymbol(&hSum, dSum, sizeof(int));
	cout << "A soma dos valores de 1 a  " << count
		<< " igual a " << hSum << " e foi processada em " << elapsed << " msec" << endl;
	getchar();

	cudaFree(d);

	return 0;
}