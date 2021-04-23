#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <device_launch_parameters.h>


typedef unsigned char uchar;

int divUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void feedCUDA_kernel(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step)
{
	int x, y, pixel, pixelOut, pixel_mask, pixelOut_mask;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height){
		return;
	}
	if (x >= width){
		return;
	}

	// MASK
	pixel_mask = (y * (mask_step)) + x;
	pixelOut_mask = ((y + dy) * (mask_dst_step)) + (x + dx);

	// DST 8 BIT
	// Get pixel index. 3 is the num of channel, dx and dy are the deltas
	//pixel = (y * (img_step)) +  3 * x;
	//pixelOut = ((y + dy) * (dst_step)) + (3*(x + dx));
	//// ------ uchar
	//const uchar img_px = img[pixel];
	//dst[pixelOut] = img[pixel];
	//dst[pixelOut + 1] = img[pixel + 1];
	//dst[pixelOut + 2] = img[pixel + 2];
	//// ------ uchar3
	////const uchar3 img_px = img[pixel];
	////dst[pixelOut] = make_uchar3(img_px.x, img_px.y, img_px.z);

	// DST 16 BIT
	// Get pixel index. 3 is the num of channel, dx and dy are the deltas
	pixel = (y * (img_step)) + 6 * x;
	pixelOut = ((y + dy) * (dst_step)) + (6 * (x + dx));
	// ------ uchar
	const uchar img_px = img[pixel];
	if (mask[pixel_mask]) {
		dst[pixelOut] = img[pixel];
		dst[pixelOut + 1] = img[pixel + 1];
		dst[pixelOut + 2] = img[pixel + 2];
		dst[pixelOut + 3] = img[pixel + 3];
		dst[pixelOut + 4] = img[pixel + 4];
		dst[pixelOut + 5] = img[pixel + 5];
	}

	// MASK
	dst_mask[pixelOut_mask] |= mask[pixel_mask] ;

	return;
}


extern "C" void feedCUDA(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim(divUp(width, blockDim.x), divUp(height, blockDim.y), 1);

	feedCUDA_kernel << <gridDim, blockDim >> >(img, mask, dst, dst_mask, dx, dy, width, height, img_step, dst_step, mask_step, mask_dst_step);


	cudaDeviceSynchronize();
}




extern "C" void feedCUDA_Async(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step, cudaStream_t streamimg, cudaStream_t streammask)
{
	

	uint blockSize = 32;	
	dim3 blockDim(blockSize, blockSize, 1);
	dim3 gridDim(divUp(width, blockDim.x), divUp(height, blockDim.y), 1);
	


	feedCUDA_kernel << <gridDim, blockDim >> >(img, mask, dst, dst_mask, dx, dy, width, height, img_step, dst_step, mask_step, mask_dst_step);

	cudaStreamAttachMemAsync(streamimg, dst, 0 , cudaMemAttachGlobal);
	cudaStreamAttachMemAsync(streammask, dst_mask, 0 , cudaMemAttachGlobal);
	
}
















