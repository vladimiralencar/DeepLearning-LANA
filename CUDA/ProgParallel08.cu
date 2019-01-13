#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define AxCheckError(err) CheckError(err,__FUNCTION__, __LINE__)
#define AxCheckErrorMsg(err, msg) CheckErrorMsg(err, msg, __FUNCTION__, __LINE__)


int const N = 1024;
int const N_BYTES = N*sizeof(float);

void GenerateTestData(int const N, float* const a, float* const b,
                      float* const ref);
void CheckError(cudaError_t const err, char const* const fun, const int line);
void CheckErrorMsg(cudaError_t const err, char const* const msg, char const* const fun, int const line);


__global__ void DotProduct(float* a, float* b, float* c)
{
    __shared__ float products[N];

    products[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

    __syncthreads();

    if(threadIdx.x == 0)
    {
        float temp = 0.0f;
        for(int i = 0; i < N; i++)
            temp += products[i];

        *c = temp;
    }
}


int main()
{
    float *aH, *bH;
    float refH;
    float cH = 0.0f;
    float *aD, *bD, *cD;

    cudaError_t e = cudaSuccess;

    dim3 gridSize;
    dim3 blockSize;

    aH = (float*)malloc(N_BYTES);
    bH = (float*)malloc(N_BYTES);

    GenerateTestData(N, aH, bH, &refH);

    e = cudaMalloc(&aD, N_BYTES);
    AxCheckError(e);
    e = cudaMalloc(&bD, N_BYTES);
    AxCheckError(e);
    e = cudaMalloc(&cD, sizeof(float));
    AxCheckError(e);

    e = cudaMemcpy(aD, aH, N_BYTES, cudaMemcpyHostToDevice);
    AxCheckError(e);
    e = cudaMemcpy(bD, bH, N_BYTES, cudaMemcpyHostToDevice);
    AxCheckError(e);

    DotProduct<<<1,N>>>(aD,bD,cD);

    e = cudaMemcpy(&cH, cD, sizeof(float), cudaMemcpyDeviceToHost);
    AxCheckError(e);

    printf("CPU: %.4f\nGPU: %.4f\n", refH, cH);

    free(aH); free(bH);
    e = cudaFree(aD); 
    AxCheckError(e);
    e = cudaFree(bD); 
    AxCheckError(e);
    e = cudaFree(cD);
    AxCheckError(e);

    AxCheckError(cudaDeviceReset());

	getchar();
    return 0;
}

void GenerateTestData(int const N, float* const a, float* const b, float* const c)
{
    int i;

    srand((unsigned)time(NULL));
    float dp = 0.0f;

    for(i = 0; i < N; i++)
    {
        a[i] = (float) rand() / RAND_MAX;
        b[i] = (float) rand() / RAND_MAX;
        dp += a[i]*b[i];
    }

    *c = dp;
}

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

