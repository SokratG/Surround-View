#include "yuv2rgb.cuh"

__host__ inline int divUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__device__ inline float clamp(float val, float min, float max)
{
	return (val >= min) ? ((val <= max) ? val : max) : min;
}



__global__ inline void gpuConvertUYVY2RGB_kernel(uchar* src, uchar* dst, uint width, uint height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx*2 >= width)
		return;

	for(int y = 0; y < height; ++y){
		const int y_w = y*width;
		int cb = src[y_w*2+idx*4];
		int y0 = src[y_w*2+idx*4+1];
		int cr = src[y_w*2+idx*4+2];
		int y1 = src[y_w*2+idx*4+3];
		
		dst[y_w*3+idx*6]   = clamp(1.164f * (y0 - 16)                       + 2.018f * (cb - 128), .0f, 255.f);
		dst[y_w*3+idx*6+1] = clamp(1.164f * (y0 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), .0f, 255.f);
		dst[y_w*3+idx*6+2] = clamp(1.164f * (y0 - 16) + 1.596f * (cr - 128),                       .0f, 255.f);
					
		dst[y_w*3+idx*6+3] = clamp(1.164f * (y1 - 16)                       + 2.018f * (cb - 128), .0f, 255.f);
		dst[y_w*3+idx*6+4] = clamp(1.164f * (y1 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), .0f, 255.f);
		dst[y_w*3+idx*6+5] = clamp(1.164f * (y1 - 16) + 1.596f * (cr - 128),                       .0f, 255.f);
	}	
}

// another example:
// https://github.com/dusty-nv/jetson-video/blob/master/cuda/cudaYUV-YUYV.cu
__global__ inline void gpuConvertUYVY2RGB_opt_kernel(uchar* src, uchar* dst, uint width, uint height)
{
      const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
      const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

      if (x*2 >= width || y >= height)
        return;


      const int y_w = y*width;

      float cb = src[y_w*2+x*4];    // U
      float y0 = src[y_w*2+x*4+1];  // Y0
      float cr = src[y_w*2+x*4+2];  // V
      float y1 = src[y_w*2+x*4+3];  // Y1

      //y0 -= 16.0f;
      //y1 -= 16.0f;
      cb -= 128.0f;
      cr -= 128.0f;

//#define YUV_I
#ifdef YUV_I

      dst[y_w*3+x*6]   = clamp(1.164f * y0 + 2.018f * cb, 0.0f, 255.f);
      dst[y_w*3+x*6+1] = clamp(1.164f * y0 - 0.813f * cr - 0.391f * cb, .0f, 255.f);
      dst[y_w*3+x*6+2] = clamp(1.164f * y0 + 1.596f * cr, 0.0f, 255.f);

      dst[y_w*3+x*6+3] = clamp(1.164f * y1 + 2.018f * cb, 0.0f, 255.f);
      dst[y_w*3+x*6+4] = clamp(1.164f * y1 - 0.813f * cr - 0.391f * cb, .0f, 255.f);
      dst[y_w*3+x*6+5] = clamp(1.164f * y1 + 1.596f * cr, 0.0f, 255.f);

#else

      dst[y_w*3+x*6]   = clamp(y0 + 1.770f * cb, 0.0f, 255.f);
      dst[y_w*3+x*6+1] = clamp(y0 - 0.344f * cb - 0.714f * cr, .0f, 255.f);
      dst[y_w*3+x*6+2] = clamp(y0 + 1.403f * cr, 0.0f, 255.f);

      dst[y_w*3+x*6+3] = clamp(y1 + 1.770f * cb, 0.0f, 255.f);
      dst[y_w*3+x*6+4] = clamp(y1 - 0.344f * cb - 0.714f * cr, .0f, 255.f);
      dst[y_w*3+x*6+5] = clamp(y1 + 1.403f * cr, 0.0f, 255.f);

#endif

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




void gpuConvertUYVY2RGB_opt(uchar* src, uchar* d_src, uchar* dst, uint width, uint height, cudaStream_t stream)
{
	size_t planeSize = width * height * sizeof(uchar);

	cudaMemcpyAsync(d_src, src, planeSize * 2, cudaMemcpyHostToDevice, stream);

	cudaStreamAttachMemAsync(stream, dst, 0 , cudaMemAttachGlobal);


	const dim3 block(16, 16);
	const dim3 grid(divUp(width/2, block.x), divUp(height, block.y));
	gpuConvertUYVY2RGB_opt_kernel <<<grid, block, 0, stream>>>(d_src, dst, width, height);

	//cudaStreamSynchronize(NULL);

}
