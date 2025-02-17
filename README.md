# cuda-python-examples
import numpy as np
from cupy import *
import time

# Define a function for matrix multiplication
def matmul_cpu(A, B):
    return dot(A, B)

def matmul_gpu(A, B):
    return dot(A, B)

# Create two random 1000x1000 matrices
n = 1000
A_cpu = np.random.rand(n, n).astype(np.float32)
B_cpu = np.random.rand(n, n).astype(np.float32)
A_gpu = cupy.array(A_cpu)
B_gpu = cupy.array(B_cpu)

# Measure the execution time of CPU matrix multiplication
start_time_cpu = time.time()
C_cpu = matmul_cpu(A_cpu, B_cpu)
end_time_cpu = time.time()

# Measure the execution time of GPU matrix multiplication
start_time_gpu = time.time()
C_gpu = matmul_gpu(A_gpu, B_gpu)
end_time_gpu = time.time()

# Print the results
print(f"CPU Execution Time: {end_time_cpu - start_time_cpu} seconds")
print(f"GPU Execution Time: {end_time_gpu - start_time_gpu} seconds")

# Check if the results are correct
assert np.allclose(cupy.asnumpy(C_gpu), C_cpu)

