#include <stdio.h>

#define NX 200
#define NY 100

__global__ void my_kernel2D(float scalar, float * x, float * y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Verifica se ainda temos threads antes de executar a operação
    if ( row < NX && col < NY ) 
        y[row * NY + col] = scalar * x[row * NY + col] + y[row * NY + col];
}

int main()
{
    float *x, *y;
    float maxError = 0;

    // Total de bytes por vetor
    int size = NX * NY * sizeof (float); 

    cudaError_t ierrAsync;
    cudaError_t ierrSync;

    cudaDeviceProp prop;

    // Aloca memória
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);

    // Inicializa memória
    for( int i = 0; i < NX*NY; ++i )
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    dim3 threads_per_block (32,16,1);
    dim3 number_of_blocks ((NX/threads_per_block.x)+1, (NY/threads_per_block.y)+1, 1);

    cudaGetDeviceProperties(&prop, 0);
    if (threads_per_block.x * threads_per_block.y * threads_per_block.z > prop.maxThreadsPerBlock) {
        printf("Muitas threads por bloco ... finalizando\n");
        goto cleanup;
    }
    if (threads_per_block.x > prop.maxThreadsDim[0]) {
        printf("Muitas threads na dimensão x ... finalizando\n");
        goto cleanup;
    }
    if (threads_per_block.y > prop.maxThreadsDim[1]) {
        printf("Muitas threads na dimensão y ... finalizando\n");
        goto cleanup;
    }
    if (threads_per_block.z > prop.maxThreadsDim[2]) {
        printf("Muitas threads na dimensão z ... finalizando\n");
        goto cleanup;
    }

    my_kernel2D <<< number_of_blocks, threads_per_block >>> ( 2.0f, x, y );

    ierrSync = cudaGetLastError();

    // Espera a GPU finalizar
    ierrAsync = cudaDeviceSynchronize(); 
    if (ierrSync != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); }
    if (ierrAsync != cudaSuccess) { printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); }

    // Imprime o erro
    for( int i = 0; i < NX*NY; ++i )
        if (abs(4-y[i]) > maxError) { maxError = abs(4-y[i]); }
    printf("Max Error: %.5f", maxError);

cleanup:
    // Libera memória alocada
    cudaFree( x ); cudaFree( y );
}