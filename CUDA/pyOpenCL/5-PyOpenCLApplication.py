# Programação Paralela com OpenCL

# Imports
from time import time  
import pyopencl as cl  
import numpy as np  
import PyOpenCLDeviceInfo as device_info

N = 150000

# Criando vetores com 10 mil posições
a = np.random.rand(N).astype(np.float32)  
b = np.random.rand(N).astype(np.float32)  

# Soma dos arrays da CPU
def cpu_array_sum(a, b):  
    c_cpu = np.empty_like(a)  
    cpu_start_time = time()  

    # Loop for para somar os valores dos arrays
    for i in range(N):
            for j in range(N):  
                    c_cpu[i] = a[i] + b[i]  
                    
    cpu_end_time = time()  

    print("\nCPU Time: {0} s".format(cpu_end_time - cpu_start_time))  
    return c_cpu  

# Soma dos arrays da GPU
def gpu_array_sum(a, b):
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[1] # GPU do Mac
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties = cl.command_queue_properties.PROFILING_ENABLE)  
    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = a)
    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = b)
    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes) 
    
    # Compila o programa para o device
    program = cl.Program(context, """
    __kernel void sum(__global const float *a, __global const float *b, __global float *c)
    {
        long N = 150000;
        int i = get_global_id(0); 
        int j;
        for(j = 0; j < N; j++)
        {
            c[i] = a[i] + b[i];
        }
    }""").build()  

    gpu_start_time = time()  
    event = program.sum(queue, a.shape, None, a_buffer, b_buffer, c_buffer)  

    # Aguarda até que o evento termine
    event.wait()  
    elapsed = 1e-9*(event.profile.end - event.profile.start)  
    print("\nGPU Kernel Time: {0} s".format(elapsed)) 
    c_gpu = np.empty_like(a) 

    # Grava os dados da memória da GPU de volta na memória do host
    cl.enqueue_read_buffer(queue, c_buffer, c_gpu).wait()  
    gpu_end_time = time()  
    print("\nGPU Time: {0} s".format(gpu_end_time - gpu_start_time))  
    print("\n")  
    return c_gpu  



if __name__ == "__main__":
    device_info.print_device_info()
    gpu_array_sum(a, b) 
    cpu_array_sum(a, b) 



