#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/types.hpp>

typedef unsigned char uchar;

__host__ inline int divUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__global__ void normalizeUsingWeightKernel32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src, const int width, const int height)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height)
    {
        constexpr float WEIGHT_EPS = 1e-5f;
        const short3 v = ((short3*)src.ptr(y))[x];
        float w = weight.ptr(y)[x];
        ((short3*)src.ptr(y))[x] = make_short3(static_cast<short>(v.x / (w + WEIGHT_EPS)),
                                               static_cast<short>(v.y / (w + WEIGHT_EPS)),
                                               static_cast<short>(v.z / (w + WEIGHT_EPS)));
    }
}


extern "C" void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
                                              const int width, const int height)
{
    dim3 threads(32, 32);
    dim3 grid(divUp(width, threads.x), divUp(height, threads.y));
    normalizeUsingWeightKernel32F<<<grid, threads>>> (weight, src, width, height);
}



extern "C" void normalizeUsingWeightMapGpu32F_Async(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
                                              const int width, const int height, cudaStream_t stream_src)
{
    cudaStreamAttachMemAsync(stream_src, src, 0 , cudaMemAttachGlobal);
    dim3 threads(32, 32);
    dim3 grid(divUp(width, threads.x), divUp(height, threads.y));
    normalizeUsingWeightKernel32F<<<grid, threads>>> (weight, src, width, height);
}


// ------------------------------- CUDABlender --------------------------------
// take from - https://github.com/Avandrea/OpenCV-BlenderGPU
__global__ void feedCUDA_kernel(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y >= height || x >= width){
		return;
	}

	// MASK
	int pixel_mask = (y * (mask_step)) + x;
	int pixelOut_mask = ((y + dy) * (mask_dst_step)) + (x + dx);

	// DST 16 BIT
	// Get pixel index. 3 is the num of channel, dx and dy are the deltas
	int pixel = (y * (img_step)) + 6 * x;
	int pixelOut = ((y + dy) * (dst_step)) + (6 * (x + dx));


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

      cudaStreamAttachMemAsync(streamimg, dst, 0 , cudaMemAttachGlobal);
      cudaStreamAttachMemAsync(streammask, dst_mask, 0 , cudaMemAttachGlobal);
      uint blockSize = 32;
      dim3 blockDim(blockSize, blockSize, 1);
      dim3 gridDim(divUp(width, blockDim.x), divUp(height, blockDim.y), 1);


      feedCUDA_kernel << <gridDim, blockDim >> >(img, mask, dst, dst_mask, dx, dy, width, height, img_step, dst_step, mask_step, mask_dst_step);
}





// ------------------------------- CUDAFeatherBlender --------------------------------
__global__ void weightBlendCUDA_kernel(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
            cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, int width, int height, int dx, int dy)
{
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x < width && y < height)
      {
          const short3 v = ((const short3*)src.ptr(y))[x];
          float w = src_weight.ptr(y)[x];
          ((short3*)dst.ptr(dy + y))[dx + x].x += static_cast<short>(v.x * w);
          ((short3*)dst.ptr(dy + y))[dx + x].y += static_cast<short>(v.y * w);
          ((short3*)dst.ptr(dy + y))[dx + x].z += static_cast<short>(v.z * w);
          dst_weight.ptr(dy + y)[dx + x] += w;
      }
}

extern "C" void weightBlendCUDA(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
    cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, const cv::Size& img_size, int dx, int dy)
{
      dim3 threads(32, 32);
      dim3 grid(divUp(img_size.width, threads.x), divUp(img_size.height, threads.y));

      weightBlendCUDA_kernel<<<grid, threads>>>(src, src_weight, dst, dst_weight, img_size.width, img_size.height, dx, dy);
}

extern "C" void weightBlendCUDA_Async(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
    cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, const cv::Size& img_size,
    int dx, int dy, cudaStream_t stream_dst)
{
      cudaStreamAttachMemAsync(stream_dst, dst, 0 , cudaMemAttachGlobal);
      cudaStreamAttachMemAsync(stream_dst, dst_weight, 0 , cudaMemAttachGlobal);

      dim3 threads(32, 32);
      dim3 grid(divUp(img_size.width, threads.x), divUp(img_size.height, threads.y));

      weightBlendCUDA_kernel<<<grid, threads>>>(src, src_weight, dst, dst_weight, img_size.width, img_size.height, dx, dy);


}





// ------------------------------- CUDAMultiBandBlender --------------------------------
__global__ void addSrcWeightKernel32F(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
           cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, int rows, int cols)
{
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;

     if (y < rows && x < cols)
     {
         const short3 v = ((const short3*)src.ptr(y))[x];
         float w = src_weight.ptr(y)[x];
         ((short3*)dst.ptr(y))[x].x += static_cast<short>(v.x * w);
         ((short3*)dst.ptr(y))[x].y += static_cast<short>(v.y * w);
         ((short3*)dst.ptr(y))[x].z += static_cast<short>(v.z * w);
         dst_weight.ptr(y)[x] += w;
     }
}


extern "C" void addSrcWeightGpu32F(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
                        cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, cv::Rect &rc)
{
     dim3 threads(16, 16);
     dim3 blocks(divUp(rc.width, threads.x), divUp(rc.height, threads.y));
     addSrcWeightKernel32F<<<blocks, threads>>>(src, src_weight, dst, dst_weight, rc.height, rc.width);
}

extern "C" void addSrcWeightGpu32F_Async(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
                        cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, cv::Rect &rc,
                        cudaStream_t stream_dst)
{
    cudaStreamAttachMemAsync(stream_dst, dst, 0 , cudaMemAttachGlobal);
    cudaStreamAttachMemAsync(stream_dst, dst_weight, 0 , cudaMemAttachGlobal);

    dim3 threads(16, 16);
    dim3 blocks(divUp(rc.width, threads.x), divUp(rc.height, threads.y));
    addSrcWeightKernel32F<<<blocks, threads>>>(src, src_weight, dst, dst_weight, rc.height, rc.width);
}





