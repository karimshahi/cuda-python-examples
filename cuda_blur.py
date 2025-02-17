import cv2
from numba import cuda, uint8
import numpy as np
import sys

kernel = np.array([[[0,0,0],[0.2,0.2,0.2],[0,0,0]],
	[[0.2,0.2,0.2],[0.2,0.2,0.2],[0.2,0.2,0.2]],
	[[0,0,0],[0.2,0.2,0.2],[0,0,0]]])

@cuda.jit
def blur_gpu(img, out, kernel):
	x, y, z = cuda.grid(3)
	
	if x >= out.shape[0] or y >= out.shape[1]:
		return

	tmp = 0
	for i in range(3):
		for j in range(3):
			tmp += kernel[i,j,z] * img[x+i,y+j,z]
	out[x,y,z] = tmp

if len(sys.argv) < 2:
	print('Usage: python cuda_blur.py path_to_img')
	sys.exit(1)

img = cv2.imread(sys.argv[1])
out = np.zeros([img.shape[0]-2, img.shape[1]-2, 3], dtype = np.uint8)

griddim = (int((out.shape[0]-1)/32) + 1, int((out.shape[1]-1)/32) + 1, 3)
blockdim = (32, 32)

blur_gpu[griddim, blockdim](img, out, kernel)

if len(sys.argv) == 3:
	name = sys.argv[2]
else:	
	name,ext = sys.argv[1].split('.')
	name += '_blur.' + ext

cv2.imwrite(name, out)

-----------------
pip install cupy
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
