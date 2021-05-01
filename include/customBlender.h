/** @brief Custom blender for gpu accelerated operations.
Simple blender which puts one image over another.
*/

#include "opencv2/core/cuda.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include <cuda_runtime.h>


class CUDABlender
{
private:
	cudaStream_t _cudaStreamImage;
	cudaStream_t _cudaStreamMask;	
public:
        CUDABlender();
        ~CUDABlender();

	/** @brief Prepares the blender for blending.

	@param corners Source images top-left corners
	@param sizes Source image sizes
	*/
	void prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes);
  
	/** @overload */
	void prepare(cv::Rect dst_roi);
  
	/** @brief Processes the image.
	@param img Source image
	@param mask Source image mask
	@param tl Source image top-left corners
	*/
        void feed(cv::cuda::GpuMat& img, cv::cuda::GpuMat& mask, cv::Point tl);
	
        /** @brief Blends and returns the final pano.
	@param dst Final pano
	@param dst_mask Final pano mask
	*/
        void blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj);

public:
	cv::cuda::GpuMat dst_, dst_mask_;
	cv::Rect dst_roi_;
};



class CUDAFeatherBlender
{
private:
        cudaStream_t _cudaStreamDst;
        cudaStream_t _cudaStreamDst_weight;
public:
        CUDAFeatherBlender(const float sharpness = 0.02f);
        ~CUDAFeatherBlender();

        void prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes);

        void prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes, const std::vector<cv::cuda::GpuMat>& masks);

        void prepare(cv::Rect dst_roi);


        void feed(cv::cuda::GpuMat& img, cv::cuda::GpuMat& mask, const cv::Point& tl, cv::cuda::Stream& streamObj);


        void blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj);

private:
        void createWeightMap(const cv::cuda::GpuMat& mask, cv::cuda::GpuMat& weight_map, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null());
public:
        cv::cuda::GpuMat dst_, dst_mask_;
        cv::cuda::GpuMat dst_weight_map_;
        cv::Rect dst_roi_;
        std::vector<cv::cuda::GpuMat> weight_maps_;
        int numBlocks_;
        int idx_weight_map_;
        float sharpness_;
        bool use_cache_weight_ = false;
};

