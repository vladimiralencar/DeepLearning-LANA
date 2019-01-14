#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
    printf("Olá! Eu sou uma thread no bloco %d\n", blockIdx.x);
}


int main(int argc,char **argv)
{
    // Inicializa o kernel
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    // Sincroniza todas as threads antes de passar o controle de volta para a CPU
    cudaDeviceSynchronize();

    printf("Processamento Concluído!\n");

    return 0;
}