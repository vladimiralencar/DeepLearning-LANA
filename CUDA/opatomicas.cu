// Fun??es at?micas ajudam a resolver o problema de ter muitas threads acessando a mesma ?rea de mem?ria
// Opera??es at?micas garantem que somente uma thread esteja acessando uma ?rea de mem?ria em um dado momento
// Opera??es at?micas devem ser configuradas com sm_20_atomic_functions.h ou o padr?o correspondente da arquitetura


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_20_atomic_functions.h"

#include <iostream>
using namespace std;

__device__ int dSum = 0;

__global__ void sum(int* d)
{
	int tid = threadIdx.x;

	// Essa instru??o vai gerar um problema, pois estamos tratando as threads como sequenciais,
	// problema conhecido como race condition
	//dSum += d[tid];

	// A fun??o atomicAdd evita o problema de race condition
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

	sum << <1, count >> >(d);

	int hSum;

	cudaMemcpyFromSymbol(&hSum, dSum, sizeof(int));

	cout << "A soma dos valores de 1 a " << count << " igual a " << hSum << endl;


	cudaFree(d);

}