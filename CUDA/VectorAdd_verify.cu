#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define AxCheckError(err) CheckError(err,__FUNCTION__, __LINE__)
#define AxCheckErrorMsg(err, msg) CheckErrorMsg(err, msg, __FUNCTION__, __LINE__)

// Gera dados de teste
void GenerateTestData(int const N, float* const a, float* const b, float* const c, float* const ref);

// Compara dados
void CompareData(int const N, float const* const a, float const* const b);

// Checa erros
void CheckError(cudaError_t const err, char const* const fun, const int line);
void CheckErrorMsg(cudaError_t const err, char const* const msg, char const* const fun, int const line);

// kernel
__global__ void SumArrays(float* const a, float* const b, float* const c, int const N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global index das threads - Total de threads

    if (i < N)
        c[i] = a[i] + b[i];
}

int main()
{
	// Vari?veis do host
    float *aH, *bH, *cH, *refH;

	// Vari?veis do device
    float *aD, *bD, *cD; // matrizes na GPU

	// CUDA error
    cudaError_t e = cudaSuccess;

	// Dimens?es
    dim3 gridSize;
    dim3 blockSize;

	// Constantes
    int const N = 2053;
    int const N_BYTES = N * sizeof(float);
    int const BLOCK_SIZE = 512; 

	// Aloca??o de mem?ria no host
    aH = (float*)malloc(N_BYTES);
    bH = (float*)malloc(N_BYTES);
    cH = (float*)malloc(N_BYTES);
    refH = (float*)malloc(N_BYTES);

	// Gera dados de teste
    GenerateTestData(N, aH, bH, cH, refH);

	// Aloca mem?ria na GPU
    e = cudaMalloc((void**)&aD, N_BYTES);
    AxCheckError(e);
    e = cudaMalloc((void**)&bD, N_BYTES);
    AxCheckError(e);
    e = cudaMalloc((void**)&cD, N_BYTES);
    AxCheckError(e);

	// Copia os valores das vari?veis na mem?ria do host para a mem?ria do device
    e = cudaMemcpy(aD, aH, N_BYTES, cudaMemcpyHostToDevice);
    AxCheckError(e);
    e = cudaMemcpy(bD, bH, N_BYTES, cudaMemcpyHostToDevice);
    AxCheckError(e);

	// Define as dimens?es
    blockSize.x = BLOCK_SIZE; blockSize.y = 1; blockSize.z = 1;
    gridSize.x = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); gridSize.y = 1; gridSize.z = 1;

	// Executa o kernel - Soma os arrays
    SumArrays<<<gridSize, blockSize>>>(aD, bD, cD, N);

    // Obt?m erros de execu??o do kernel
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    AxCheckError(e);

	// Copia o resultado da mem?ria do device para a mem?ria do host
    e = cudaMemcpy(cH, cD, N_BYTES, cudaMemcpyDeviceToHost);
    AxCheckError(e);

	// Compara os dados
    CompareData(N, cH, refH);

	// Libera a mem?ria
    cudaFree(aD); cudaFree(bD); cudaFree(cD);
    free(aH); free(bH); free(cH); free(refH);
    AxCheckError(cudaDeviceReset());


    return 0;
}

// Fun??o para gerar os dados - CPU
void GenerateTestData(int const N, float* const a, float* const b, float* const c, float* const ref)
{
    int i;
	
    srand((unsigned)time(NULL));

    for(i = 0; i < N; i++)
    {
        a[i] = (float)rand();
        b[i] = (float)rand();
        c[i] = 0.0f;
        ref[i] = a[i] + b[i];
    }

}

// Fun??o para comparar os dados
void CompareData(int const N, float const* const a, float const* const b)
{
    int i;
    int different = 0;

    for(i = 0; i < N; i++)
    {
        different = (a[i] != b[i]);
        if(different)
            break;
    }

    if(different)
    {
        printf("Arrays diferentes.\n");
    }
    else
    {
        printf("Arrays match.\n");
    }

}

// Fun??es para checar e imprimir o c?digo de erro
void CheckError(cudaError_t const err, char const* const fun, const int line)
{
    if (err)
    {
        printf("CUDA Error Code[%d]: %s %s():%d\n",err,cudaGetErrorString(err),fun,line);
        exit(1);
    }
}

void CheckErrorMsg(cudaError_t const err, char const* const msg, char const* const fun, int const line)
{
    if (err)
    {
        printf("CUDA Error Code[%d]: %s %s() %d\n%s\n",err,cudaGetErrorString(err),fun,line,msg);
        exit(1);
    }
}
