#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <numeric>
using namespace std;

__global__ void sumSingleBlock(int* d)
{
	int tid = threadIdx.x;

	// N?mero de threads participando em cada itera??o 
	for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1)
	{
		// Thread deve ter permiss?o para escrever. Thread ID deve ser menor que Thread Count
		if (tid < tc)
		{
			// Definindo como ser? a opera??o de Reduce
			// Precisamos especificar que a thread obteve o resultado e ent?o somar com o pr?ximo elemento do array
			int pa = tid * stepSize * 2;

			// Obtemos o que foi escrito (gravado) pela thread e somamos com o pr?ximo elemento do array
			int pb = pa + stepSize;
			d[pa] += d[pb];
		}
	}
}


int main()
{
	// Status de erro
	cudaError_t status;

	// Constantes
	const int count = 256;
	const int size = count * sizeof(int);

	// Definindo um array de valores inteiros 
	// Array no host
	int* h = new int[count];

	// Preenchendo o array com elementos
	for (int i = 0; i < count; ++i)
		h[i] = i + 1;

	// Array no device
	int* d;

	// Alocando device memory
	status = cudaMalloc(&d, size);

	// Copiando da mem?ria RAM para a mem?ria do device
	status = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	// Um bloco de thread e o m?ximo de threads poss?vel em nosso caso, count/2
	sumSingleBlock << <1, count / 2, size >> >(d);

	int result;

	// Devolvendo os elementos do device para a mem?ria RAM
	status = cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Soma dos Elementos do array igual a " << result << endl;

	cudaFree(d);
	delete[] h;

	return 0;
}

