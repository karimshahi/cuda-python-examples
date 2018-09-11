import cv2
from numba import cuda
import sys
import numpy as np

# Threads per block
tpb = 32

@cuda.autojit
def rgb2gray(d_img,d_gray):
	x, y = cuda.grid(2)
	if x >= d_img.shape[0] or y >= d_img.shape[1]:
		return
	d_gray[x,y,0] = d_img[x,y,2] * 0.3 + d_img[x,y,1] * 0.59 + d_img[x,y,0] * 0.11

if len(sys.argv) < 2:
	print('Usage: python rgb2gray.py path_to_img')
	sys.exit(1)

img = cv2.imread(sys.argv[1])
gray = np.zeros([img.shape[0],img.shape[1],1],dtype=np.uint8)

blockdim = (tpb, tpb)
griddim = (int(img.shape[0]-1/tpb) + 1, int(img.shape[1]-1/tpb) + 1)

rgb2gray[griddim,blockdim](img,gray)

if len(sys.argv) == 3:
	name = sys.argv[2]
else:	
	name,ext = sys.argv[1].split('.')
	name += '_gray.' + ext

cv2.imwrite(name,gray)
