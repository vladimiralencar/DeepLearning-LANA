
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addArraysGPU(int* a, int* b, int* c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int  main()
{
    // Constante
    const int count = 5;
    const int size = count * sizeof(int);

    // Arrays - Memria RAM
    int ha[] = { 1, 2, 3, 4, 5 };
    int hb[] = { 100, 200, 300, 400, 500 };

    // Array para gravar o resultado - Memria RAM
    int hc[count];

    // Variveis para execuo na GPU
    int *da, *db, *dc;

    // Alocao de memria na GPU
    cudaMalloc(&da, size); // Memory Allocation
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    // Cpia das variveis a e b da Memria RAM para a Memria na GPU
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    // Definindo um bloco de threads - 1 bloco - de count=5 threads
    addArraysGPU <<<1, count >>>(da, db, dc);

    // Cpia do resultado da Memria da GPU de volta para a Memria da CPU
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    // Imprime os resultados
    printf("%d %d %d %d %d",
        hc[0],
        hc[1],
        hc[2],
        hc[3],
        hc[4]);

    // Libera as reas de memria
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    // Para visualizar o resultado na tela ate pressionar uma tecla
    getchar();

}

