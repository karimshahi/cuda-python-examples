import numpy as np
from numba import cuda, float32

TPB = 32 # Threads Per Block

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(hid):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

# Define the matrices
a = np.random.randn(6400, 1024)
b = np.random.randn(1024, 2048)
c = np.zeros([6400,2048])

-------------------


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


hid = int(1023/32) + 1

blockdim = (TPB, TPB)
griddim = (int((c.shape[0]-1)/TPB) + 1, int((c.shape[1]-1)/TPB) + 1)

fast_matmul[griddim,blockdim](a,b,c)
