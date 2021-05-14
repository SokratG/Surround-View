#include "yuv2rgb.cuh"

__device__ inline float clamp(float val, float min, float max)
{
	return (val >= min) ? ((val <= max) ? val : max) : min;
}


/*
__global__ inline void gpuConvertUYVY2RGB_kernel(uchar* src, uchar* dst, uint width, uint height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx*2 >= width)
		return;

	for(int i=0; i<height; ++i){
		int cb = src[i*width*2+idx*4];
		int y0 = src[i*width*2+idx*4+1];
		int cr = src[i*width*2+idx*4+2];
		int y1 = src[i*width*2+idx*4+3];
		
		dst[i*width*3+idx*6]   = clamp(1.164f * (y0 - 16) + 1.596f * (cr - 128),                       .0f, 255.f);	
		dst[i*width*3+idx*6+1] = clamp(1.164f * (y0 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), .0f, 255.f);
		dst[i*width*3+idx*6+2] = clamp(1.164f * (y0 - 16)                       + 2.018f * (cb - 128), .0f, 255.f);	
					
		dst[i*width*3+idx*6+3] = clamp(1.164f * (y1 - 16) + 1.596f * (cr - 128),                       .0f, 255.f);
		dst[i*width*3+idx*6+4] = clamp(1.164f * (y1 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), .0f, 255.f);
		dst[i*width*3+idx*6+5] = clamp(1.164f * (y1 - 16)                       + 2.018f * (cb - 128), .0f, 255.f);
	}
	
}*/

__global__ inline void gpuConvertUYVY2RGB_kernel(uchar* src, uchar* dst, uint width, uint height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx*2 >= width)
		return;

	for(int y = 0; y < height; ++y){
		int cb = src[y*width*2+idx*4];
		int y0 = src[y*width*2+idx*4+1];
		int cr = src[y*width*2+idx*4+2];
		int y1 = src[y*width*2+idx*4+3];
		
		dst[y*width*3+idx*6]   = clamp(1.164f * (y0 - 16)                       + 2.018f * (cb - 128), .0f, 255.f);
		dst[y*width*3+idx*6+1] = clamp(1.164f * (y0 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), .0f, 255.f);
		dst[y*width*3+idx*6+2] = clamp(1.164f * (y0 - 16) + 1.596f * (cr - 128),                       .0f, 255.f);
					
		dst[y*width*3+idx*6+3] = clamp(1.164f * (y1 - 16)                       + 2.018f * (cb - 128), .0f, 255.f);
		dst[y*width*3+idx*6+4] = clamp(1.164f * (y1 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), .0f, 255.f);
		dst[y*width*3+idx*6+5] = clamp(1.164f * (y1 - 16) + 1.596f * (cr - 128),                       .0f, 255.f);
	}
	
}


void gpuConvertUYVY2RGB(uchar* src, uchar* dst, uint width, uint height)
{

	uchar* d_src = NULL;
	uchar* d_dst = NULL;
	size_t planeSize = width * height * sizeof(uchar);

	uint flags;
	bool srcIsMapped = (cudaHostGetFlags(&flags, src) == cudaSuccess) && (flags & cudaHostAllocMapped);
	bool dstIsMapped = (cudaHostGetFlags(&flags, dst) == cudaSuccess) && (flags & cudaHostAllocMapped);

	if (srcIsMapped){
		d_src = src;
		cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);
	}
	else{
		cudaMalloc(&d_src, planeSize * 2);
		cudaMemcpy(d_src, src, planeSize * 2, cudaMemcpyHostToDevice);
	}
	if (dstIsMapped){
		d_dst = dst;
		cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);
	}
	else
		cudaMalloc(&d_dst, planeSize * 3);


	uint blockSize = 1024;
	uint numBlocks = (width / 2 + blockSize - 1) / blockSize;
	gpuConvertUYVY2RGB_kernel <<<numBlocks, blockSize >>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0 , cudaMemAttachHost); // Host?
	cudaStreamSynchronize(NULL);
	if (!srcIsMapped){
		cudaMemcpy(dst, d_dst, planeSize*3, cudaMemcpyDeviceToHost);
		cudaFree(d_src);
	}
	if (!dstIsMapped)
		cudaFree(d_dst);

}



void gpuConvertUYVY2RGB_async(uchar* src, uchar* dst, uint width, uint height, cudaStream_t stream)
{

	uchar* d_src = NULL;
	uchar* d_dst = NULL;
	size_t planeSize = width * height * sizeof(uchar);

	uint flags;
	bool srcIsMapped = (cudaHostGetFlags(&flags, src) == cudaSuccess) && (flags & cudaHostAllocMapped);
	bool dstIsMapped = (cudaHostGetFlags(&flags, dst) == cudaSuccess) && (flags & cudaHostAllocMapped);

	if (srcIsMapped){
		d_src = src;
		cudaStreamAttachMemAsync(stream, src, 0, cudaMemAttachGlobal);
	}
	else{
		cudaMalloc(&d_src, planeSize * 2);
		cudaMemcpy(d_src, src, planeSize * 2, cudaMemcpyHostToDevice);
	}
	if (dstIsMapped){
		d_dst = dst;
		cudaStreamAttachMemAsync(stream, dst, 0, cudaMemAttachGlobal);
	}
	else
		cudaMalloc(&d_dst, planeSize * 3);

	uint blockSize = 1024;
	uint numBlocks = (width / 2 + blockSize - 1) / blockSize;
	gpuConvertUYVY2RGB_kernel <<<numBlocks, blockSize >>>(d_src, d_dst, width, height);

	cudaStreamAttachMemAsync(stream, dst, 0 , cudaMemAttachGlobal);
	//cudaStreamSynchronize(NULL);

	if (!srcIsMapped){
		cudaMemcpy(dst, d_dst, planeSize*3, cudaMemcpyDeviceToHost);
		cudaFree(d_src);
	}
	if (!dstIsMapped)
		cudaFree(d_dst);

}



