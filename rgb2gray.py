import cv2
from numba import cuda
import sys
import numpy as np
import argparse

# Define constants for image processing
TPB = 32  # Threads per block
DEVICE_MEMORY = 4 * 1024 * 1024  # Device memory in bytes (approximately 4 MB)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Convert RGB images to grayscale using CUDA.')
    parser.add_argument('input_file', help='Path to input image file')
    parser.add_argument('-o', '--output_file', help='Output file name (default: same as input file with _gray appended)')
    return parser.parse_args()

def rgb2gray_kernel(d_img, d_gray):
    """CUDA kernel for converting RGB images to grayscale."""
    x, y = cuda.grid(2)
    if x >= d_img.shape[0] or y >= d_img.shape[1]:
        return
    d_gray[x,y,0] = d_img[x,y,2] * 0.3 + d_img[x,y,1] * 0.59 + d_img[x,y,0] * 0.11

def convert_rgb_to_grayscale(input_file, output_file):
    """Convert RGB image to grayscale using CUDA."""
    # Read input image
    img = cv2.imread(input_file)
    
    if img is None:
        print(f"Error: Unable to read file '{input_file}'")
        sys.exit(1)

    # Create device arrays for GPU processing
    d_img = cuda.device_array(img.shape, dtype=np.uint8)
    d_gray = cuda.device_array((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Copy input image to device array
    cuda.synchronize()
    d_img[:] = img

    # Launch CUDA kernel for grayscale conversion
    blockdim = (TPB, TPB)
    griddim = (int((img.shape[0]-1)/TPB) + 1, int((img.shape[1]-1)/TPB) + 1)

    rgb2gray_kernel[griddim, blockdim](d_img, d_gray)

    # Copy results back to host
    gray = np.copy(d_gray)

    # Save output image file (optional)
    if output_file:
        cv2.imwrite(output_file, gray)
    else:
        name, ext = input_file.split('.')
        name += '_gray.' + ext
        cv2.imwrite(name, gray)

if __name__ == '__main__':
    args = parse_args()
    convert_rgb_to_grayscale(args.input_file, args.output_file)
