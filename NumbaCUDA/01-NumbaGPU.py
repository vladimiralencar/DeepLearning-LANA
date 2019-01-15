# Numba Python

import numba
print("Versão do Numba")
print(numba.__version__)

from numba import cuda
import numpy as np

@cuda.jit
def max_example(result, values):
    """Encontre o valor máximo e armazene em result[0]"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    cuda.atomic.max(result, 0, values[i])


arr = np.random.rand(16384)
result = np.zeros(1, dtype = np.float64)

max_example[256,64](result, arr)

print("\nResultado encontrado com processamento na GPU:")
print(result[0]) 

print("\nResultado encontrado com processamento na CPU:")
print(max(arr))  
print(arr.shape)

print("\n")