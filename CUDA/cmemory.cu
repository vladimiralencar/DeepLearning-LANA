#include <cuda_profiler_api.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#define AxCheckError(err) CheckError(err,__FUNCTION__, __LINE__)
#define AxCheckErrorMsg(err, msg) CheckErrorMsg(err, msg, __FUNCTION__, __LINE__)

void GenerateTestData(int const N, float* const input, float* const filtered, 
                      float* const ref);
void CompareData(int const N, float const* const a, float const* const b);

void CheckError(cudaError_t const err, char const* const fun, const int line);
void CheckErrorMsg(cudaError_t const err, char const* const msg, char const* const fun, int const line);

#define BLOCK_SIZE  512


float const FILTER_COEFFS[21] = {0.005f,0.01f, 0.02f, 0.03f, 0.04f, 
                                 0.05f, 0.06f, 0.07f, 0.25f, 0.75f, 
                                 1.0f,  0.75f, 0.25f, 0.07f, 0.06f, 
                                 0.05f, 0.04f, 0.03f, 0.02f, 0.01f, 0.005f};

// Armazenado na Constant Memory
__device__ __constant__ float FilterCoeffs[21] =  {0.005f,0.01f, 0.02f, 0.03f, 0.04f, 
                                                   0.05f, 0.06f, 0.07f, 0.25f, 0.75f, 
                                                   1.0f,  0.75f, 0.25f, 0.07f, 0.06f, 
                                                   0.05f, 0.04f, 0.03f, 0.02f, 0.01f, 0.005f};


// Usa apenas a Global Memory
__global__ void GlobalFilter(float* const input, float* const filtered, int const N)
{
   
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (10 < gIdx && gIdx < N - 10)
    {
        float sum;
        sum = input[gIdx - 10] * FilterCoeffs[ 0] +
              input[gIdx -  9] * FilterCoeffs[ 1] + 
              input[gIdx -  8] * FilterCoeffs[ 2] + 
              input[gIdx -  7] * FilterCoeffs[ 3] +
              input[gIdx -  6] * FilterCoeffs[ 4] +
              input[gIdx -  5] * FilterCoeffs[ 5] + 
              input[gIdx -  4] * FilterCoeffs[ 6] + 
              input[gIdx -  3] * FilterCoeffs[ 7] +
              input[gIdx -  2] * FilterCoeffs[ 8] + 
              input[gIdx -  1] * FilterCoeffs[ 9] + 
              input[gIdx     ] * FilterCoeffs[10] + 
              input[gIdx +  1] * FilterCoeffs[11] + 
              input[gIdx +  2] * FilterCoeffs[12] +
              input[gIdx +  3] * FilterCoeffs[13] +
              input[gIdx +  4] * FilterCoeffs[14] +
              input[gIdx +  5] * FilterCoeffs[15] +
              input[gIdx +  6] * FilterCoeffs[16] +
              input[gIdx +  7] * FilterCoeffs[17] +
              input[gIdx +  8] * FilterCoeffs[18] +
              input[gIdx +  9] * FilterCoeffs[19] +
              input[gIdx + 10] * FilterCoeffs[20];

        filtered[gIdx] = sum;
    }
}

// Usa a Shared Memory
__global__ void SharedFilter(float* const input, float* const filtered, int const N)
{
    __shared__ float inputS[BLOCK_SIZE+20];
    int sIdx = threadIdx.x;
    long long gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
	// Dez valores extras no índice
    int sIdxShift = sIdx + 10;

    // Todas as threads fazem a leitura de um elemento na Global Memory e armazenam na Shared Memory.
    if (gIdx < N)
    {
        inputS[sIdxShift] = input[gIdx];
    }

    // As primeiras 10 threads no bloco armazenam os 10 valores extras nos 10 primeiros elementos da Shared Memory
    if(sIdx < 10 && blockIdx.x != 0)
    {
        inputS[sIdx] = input[gIdx - 10];
    }

    // As últimas 10 threads armazenam os 10 valores extras na Shared Memory 
    if(sIdxShift >= blockDim.x && blockIdx.x < gridDim.x - 1)
    {
        inputS[sIdxShift + 10] = input[gIdx + 10];
    }

    __syncthreads();
    
    float sum;
    sum = inputS[sIdxShift - 10] * FilterCoeffs[ 0] +
          inputS[sIdxShift -  9] * FilterCoeffs[ 1] + 
          inputS[sIdxShift -  8] * FilterCoeffs[ 2] + 
          inputS[sIdxShift -  7] * FilterCoeffs[ 3] +
          inputS[sIdxShift -  6] * FilterCoeffs[ 4] +
          inputS[sIdxShift -  5] * FilterCoeffs[ 5] + 
          inputS[sIdxShift -  4] * FilterCoeffs[ 6] + 
          inputS[sIdxShift -  3] * FilterCoeffs[ 7] +
          inputS[sIdxShift -  2] * FilterCoeffs[ 8] + 
          inputS[sIdxShift -  1] * FilterCoeffs[ 9] + 
          inputS[sIdxShift     ] * FilterCoeffs[10] + 
          inputS[sIdxShift +  1] * FilterCoeffs[11] + 
          inputS[sIdxShift +  2] * FilterCoeffs[12] +
          inputS[sIdxShift +  3] * FilterCoeffs[13] +
          inputS[sIdxShift +  4] * FilterCoeffs[14] +
          inputS[sIdxShift +  5] * FilterCoeffs[15] +
          inputS[sIdxShift +  6] * FilterCoeffs[16] +
          inputS[sIdxShift +  7] * FilterCoeffs[17] +
          inputS[sIdxShift +  8] * FilterCoeffs[18] +
          inputS[sIdxShift +  9] * FilterCoeffs[19] +
          inputS[sIdxShift + 10] * FilterCoeffs[20];

    filtered[gIdx] = sum;
}

int main()
{
    float *inputH, *filteredH, *refH;
    float *inputD, *filteredD;
    cudaError_t e = cudaSuccess;
    dim3 gridSize, gridSize2;
    dim3 blockSize;

    int const N = 16*1024*1024;
    int const N_BYTES = N * sizeof(float);

    inputH =    (float*)malloc(N_BYTES);
    filteredH = (float*)malloc(N_BYTES);
    refH =      (float*)malloc(N_BYTES);

    GenerateTestData(N, inputH, filteredH, refH);

    e = cudaMalloc((void**)&inputD, N_BYTES);
    AxCheckError(e);
    e = cudaMalloc((void**)&filteredD, N_BYTES);
    AxCheckError(e);

    e = cudaMemcpy(inputD, inputH, N_BYTES, cudaMemcpyHostToDevice);
    AxCheckError(e);

    gridSize.x = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    blockSize.x = BLOCK_SIZE; 

    int const TRIALS = 5;
    std::vector<float> sharedTimes;
    std::vector<float> globalTimes;
    cudaEvent_t start, stop;

    e = cudaEventCreate(&start);
    AxCheckError(e);
    e = cudaEventCreate(&stop);
    AxCheckError(e);

    e = cudaProfilerStart();

    for(int i = 0; i < TRIALS; i++)
    {
        e = cudaEventRecord(start, 0);
        SharedFilter<<<gridSize, blockSize>>>(inputD, filteredD, N);
        e = cudaEventRecord(stop, 0);
        AxCheckError(cudaDeviceSynchronize());
        AxCheckError(cudaGetLastError());

        float elapsed;
        e = cudaEventElapsedTime(&elapsed, start, stop);
        sharedTimes.push_back(elapsed);

        e = cudaEventRecord(start, 0);
        GlobalFilter<<<gridSize, blockSize>>>(inputD, filteredD, N);
        e = cudaEventRecord(stop, 0);
        AxCheckError(cudaDeviceSynchronize());
        AxCheckError(cudaGetLastError());

        e = cudaEventElapsedTime(&elapsed, start, stop);
        globalTimes.push_back(elapsed);
    }

    e = cudaProfilerStop();

    float averageTime = std::accumulate(globalTimes.begin(), globalTimes.end(), 0.0f)/globalTimes.size();
    std::cout << "Global Memory time (ms): " << averageTime << std::endl;
    averageTime = std::accumulate(sharedTimes.begin(), sharedTimes.end(), 0.0f)/sharedTimes.size();
    std::cout << "Shared Memory time (ms): " << averageTime << std::endl;

    /* Executando o kernel */
    SharedFilter<<<gridSize, blockSize>>>(inputD, filteredD, N);
    AxCheckError(cudaDeviceSynchronize());
    AxCheckError(cudaGetLastError());

    /* Não geramos zeros para os 10 primeiros / últimos 10 elementos no kernel. Na verdade, geramos valores usando
        Shared Memory não inicializada como entradas, logo elas estão incorretas. Portanto, não os copiamos e confiamos
        no fato de que o filtroH foi previamente ajustado para zero. */
    e = cudaMemcpy(filteredH + 10, filteredD + 10, N_BYTES - 20 * sizeof(float), cudaMemcpyDeviceToHost);
    AxCheckError(e);

    std::cout << "Validando o output do SharedFilter..." << std::endl;
    CompareData(N, filteredH, refH);

    /* Executando o kernel */
    GlobalFilter<<<gridSize, blockSize>>>(inputD, filteredD, N);
    AxCheckError(cudaDeviceSynchronize());
    AxCheckError(cudaGetLastError());

    /* Nós não geramos saída para os 10 primeiros / últimos 10 elementos no kernel. Portanto, não os copiamos e confiamos
     no fato de que o filtroH foi previamente ajustado para zero. */
    e = cudaMemcpy(filteredH + 10, filteredD + 10, N_BYTES - 20 * sizeof(float), cudaMemcpyDeviceToHost);
    AxCheckError(e);

    std::cout << "Validando o output do GlobalFilter..." << std::endl;
    CompareData(N, filteredH, refH);

    cudaFree(inputD); cudaFree(filteredD);
    free(inputH); free(filteredH); free(refH);

    AxCheckError(cudaDeviceReset());

	getchar();

    return 0;
}

void GenerateTestData(int const N, float* const input, float* const filtered, float* const ref)
{
    int i;

    for(i = 0; i < N; i++)
    {
        //input[i] = ((float)rand())/RAND_MAX;
        input[i] = i;
        filtered[i] = 0.0f;
    }

    memset(ref, 0, N*sizeof(float) );

    /* Não podemos calcular um filtro de 21 pontos nas bordas da nossa matriz.
        Se todos os 21 pontos não estiverem disponíveis, o resultado esperado é zero! */
    for(i = 10; i < N-10; i++)
    {   
        ref[i] = (input[i-10]*FILTER_COEFFS[ 0] +
                  input[i- 9]*FILTER_COEFFS[ 1] +
                  input[i- 8]*FILTER_COEFFS[ 2] +
                  input[i- 7]*FILTER_COEFFS[ 3] +
                  input[i- 6]*FILTER_COEFFS[ 4] +
                  input[i- 5]*FILTER_COEFFS[ 5] +
                  input[i- 4]*FILTER_COEFFS[ 6] +
                  input[i- 3]*FILTER_COEFFS[ 7] +
                  input[i- 2]*FILTER_COEFFS[ 8] +
                  input[i- 1]*FILTER_COEFFS[ 9] + 
                  input[i   ]*FILTER_COEFFS[10] + 
                  input[i+ 1]*FILTER_COEFFS[11] + 
                  input[i+ 2]*FILTER_COEFFS[12] + 
                  input[i+ 3]*FILTER_COEFFS[13] + 
                  input[i+ 4]*FILTER_COEFFS[14] + 
                  input[i+ 5]*FILTER_COEFFS[15] + 
                  input[i+ 6]*FILTER_COEFFS[16] + 
                  input[i+ 7]*FILTER_COEFFS[17] + 
                  input[i+ 8]*FILTER_COEFFS[18] + 
                  input[i+ 9]*FILTER_COEFFS[19] + 
                  input[i+10]*FILTER_COEFFS[20]);
    }
}

int UlpDifference(float a, float b)
{
    int iA, iB;
    iA = *((int*)(&a));
    iB = *((int*)(&b));
    return abs(iA - iB);
}

void CompareData(int const N, float const* const a, float const* const b)
{
    int i;
    int different = 0;

    for(i = 0; i < N; i++)
    {
        different = (UlpDifference(a[i],b[i]) > 5);
        if(different)
        {
            std::cout << "Mismatch: " << a[i] << " " << b[i] << std::endl;
            break;
        }
    }

    if(different)
    {
        printf("Arrays do not match @%d.\n", i);
    }
    else
    {
        printf("Arrays match.\n");
    }
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
