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

griddim = (int((out.shape[0]-1)/18) + 1, int((out.shape[1]-1)/18) + 1, 1)
blockdim = (18, 18, 3)

blur_gpu[griddim, blockdim](img, out, kernel)

if len(sys.argv) == 3:
	name = sys.argv[2]
else:	
	name,ext = sys.argv[1].split('.')
	name += '_blur.' + ext

cv2.imwrite(name, out)