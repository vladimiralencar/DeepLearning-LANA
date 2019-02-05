#include <stdio.h>

// Número de elementos em cada vetor
#define N 2048 * 2048

__global__ void my_kernel(float scalar, float * x, float * y)
{
    // Determina a identificação de thread global exclusiva, por isso sabemos qual elemento processar
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Certifique-se de que ainda temos threads disponíveis!
    if ( tid < N ) 
        y[tid] = scalar * x[tid] + y[tid];
}

int main()
{
    float *x, *y;

    // O número total de bytes por vetor
    int size = N * sizeof (float); 

    cudaError_t ierrAsync;
    cudaError_t ierrSync;

    // Aloca memória
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);

    // Inicializa a memória
    for( int i = 0; i < N; ++i )
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int threads_per_block = 256;
    int number_of_blocks = (N / threads_per_block) + 1;

    my_kernel <<< number_of_blocks, threads_per_block >>> ( 2.0f, x, y );

    ierrSync = cudaGetLastError();

    // Aguarde até que a GPU termine
    ierrAsync = cudaDeviceSynchronize(); 

    // Verifica status de execução
    if (ierrSync != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); }
    if (ierrAsync != cudaSuccess) { printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); }

    // Imprime o erro máximo
    float maxError = 0;
    for( int i = 0; i < N; ++i )
        if (abs(4-y[i]) > maxError) { maxError = abs(4-y[i]); }
    printf("Max Error: %.5f", maxError);

    // Libera a memória alocada
    cudaFree( x ); cudaFree( y );
}