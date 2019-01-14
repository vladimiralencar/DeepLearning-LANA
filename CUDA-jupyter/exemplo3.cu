#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
using namespace std;

#define N 756

// kernel
__global__ void matrixMulGPU( int * a, int * b, int * c )
{
    int val = 0;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        for ( int k = 0; k < N; ++k )
            val += a[row * N + k] * b[k * N + col];
        c[row * N + col] = val;
    }
}

void matrixMulCPU( int * a, int * b, int * c )
{
    int val = 0;

    for( int row = 0; row < N; ++row )
        for( int col = 0; col < N; ++col )
        {
            val = 0;
            for ( int k = 0; k < N; ++k )
                val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
}

int main()
{
    int *a, *b, *c_cpu, *c_gpu;

    // Número de bytes de uma matriz N x N 
    int size = N * N * sizeof (int); 

    // Aloca memória
    cudaMallocManaged (&a, size);
    cudaMallocManaged (&b, size);
    cudaMallocManaged (&c_cpu, size);
    cudaMallocManaged (&c_gpu, size);

    // Inicializa memória
    for( int row = 0; row < N; ++row )
        for( int col = 0; col < N; ++col )
        {
            a[row * N + col] = row;
            b[row * N + col] = col+2;
            c_cpu[row * N + col] = 0;
            c_gpu[row * N + col] = 0;
        }

    // Bloco de threads 16 x 16     
    dim3 threads_per_block (16, 16, 1); 
    dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

    // Define 2 eventos CUDA
    cudaEvent_t start, end;

    // Cria os eventos
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Registra o primeiro evento
    cudaEventRecord(start);

    // Chamada ao kernel
    matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

    // Registra o segundo evento
    cudaEventRecord(end);

    // Aguarda a GPU finalizar seu trabalho
    cudaDeviceSynchronize(); 

    // Calcula o tempo usado no processamento
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);

    cout << "Tempo de processamento na GPU igual a " << elapsed << " msec (aproximadamente 0.01108 segundos)" << endl;

    clock_t start1, end1;
    double cpu_time_used;

    start1 = clock();

    // Chama a versão para CPU para checar nosso trabalho
    matrixMulCPU( a, b, c_cpu );

    // Calcula o tempo usado no processamento
    end1 = clock();
    cpu_time_used = ((double) (end1 - start1)) / CLOCKS_PER_SEC;

    cout << "Tempo de processamento na CPU igual a " << cpu_time_used << " sec" << endl;

    // Compara as duas respostas para garantir que elas sejam iguais
    bool error = false;
    for( int row = 0; row < N && !error; ++row )
        for( int col = 0; col < N && !error; ++col )
            if (c_cpu[row * N + col] != c_gpu[row * N + col])
            {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error)
        printf("Successo! As duas matrizes são iguais, sendo executadas na CPU e na GPU!\n");

    // Libera a memória
    cudaFree(a); 
    cudaFree(b);
    cudaFree( c_cpu ); 
    cudaFree( c_gpu );
}