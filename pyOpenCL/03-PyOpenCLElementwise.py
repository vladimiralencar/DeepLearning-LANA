# Operações Element-Wise

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np  

# Inicializando o contexto através da escolha no console
context = cl.create_some_context() 

# Inicicializando a fila
fila = cl.CommandQueue(context)  

# Vetor
vector_dimension = 100

# Copiando os vetores para o device
vector_a = cl_array.to_device(fila,  np.random.randint(vector_dimension, size=vector_dimension))
vector_b = cl_array.to_device(fila,  np.random.randint(vector_dimension, size=vector_dimension))  

# Resultado
result_vector = cl_array.empty_like(vector_a)  

# Executando as operações element-wise
elementwiseSum = cl.elementwise.ElementwiseKernel(context, "int *a, int *b, int *c", "c[i] = a[i] + b[i]", "sum")
elementwiseSum(vector_a, vector_b, result_vector)  

print ("\nPyOpenCL - Soma Element Wise Entre 2 Vetores")
print ("Comprimento do Vetor = %s" %vector_dimension)
print ("Vetor A")
print (vector_a)
print ("Vetor B")
print (vector_b)
print ("Vetor Resultante de A + B ")
print (result_vector)
