#include "yuv2rgb.cuh"

__host__ inline int divUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__device__ inline float clamp(float val, float min, float max)
{
	return (val >= min) ? ((val <= max) ? val : max) : min;
}



// another example:
// https://github.com/dusty-nv/jetson-video/blob/master/cuda/cudaYUV-YUYV.cu
__global__ inline void gpuConvertUYVY2RGB_kernel(uchar* src, uchar* dst, uint width, uint height)
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

//#define YUV_YCbCr
#ifdef YUV_YCbCr

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




void gpuConvertUYVY2RGB_async(uchar* src, uchar* d_src, uchar* dst, uint width, uint height, cudaStream_t stream)
{
	size_t planeSize = width * height * sizeof(uchar);

	cudaMemcpyAsync(d_src, src, planeSize * 2, cudaMemcpyHostToDevice, stream);

	const dim3 block(16, 16);
	const dim3 grid(divUp(width/2, block.x), divUp(height, block.y));
	gpuConvertUYVY2RGB_kernel <<<grid, block, 0, stream>>>(d_src, dst, width, height);

	//cudaStreamSynchronize(NULL);

}
