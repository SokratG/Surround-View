#include "customBlender.h"
#include <opencv2/cudaarithm.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>


typedef unsigned char uchar;

extern "C" {
	void feedCUDA(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step);
	void feedCUDA_Async(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step, cudaStream_t streamimage, cudaStream_t streammask);
	extern "C" void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
						      const int width, const int height);
	void weightBlendCUDA(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
	    cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, const cv::Size& img_size, int dx, int dy);
}

static constexpr float WEIGHT_EPS = 1e-5f;


CUDABlender::CUDABlender(float sharpness) : sharpness_(sharpness)
{
	if (cudaStreamCreate(&_cudaStreamImage) != cudaError::cudaSuccess)
		_cudaStreamImage = NULL;
	if (cudaStreamCreate(&_cudaStreamMask) != cudaError::cudaSuccess)
		_cudaStreamMask = NULL;
}


CUDABlender::~CUDABlender(void)
{
	
}


void CUDABlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes)
{
	prepare(cv::detail::resultRoi(corners, sizes));
}


void CUDABlender::prepare(cv::Rect dst_roi)
{
	dst_ = cv::cuda::GpuMat(dst_roi.size(), CV_16SC3);
	dst_.setTo(cv::Scalar::all(0));
	dst_mask_ = cv::cuda::GpuMat(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));
	dst_roi_ = dst_roi;

	dst_weight_map_.create(dst_roi.size(), CV_32F);
	dst_weight_map_.setTo(cv::Scalar::all(0));
}


void CUDABlender::feed(cv::cuda::GpuMat& _img, cv::cuda::GpuMat& _mask, cv::Point tl)
{

	CV_Assert(_img.type() == CV_16SC3);
	CV_Assert(_mask.type() == CV_8U);
	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;

	if (_cudaStreamImage && _cudaStreamMask)
		feedCUDA_Async((uchar*)_img.data, (uchar*)_mask.data, (uchar*)dst_.data, (uchar*)dst_mask_.data, dx, dy, _img.cols, _img.rows, _img.step, dst_.step, _mask.step, dst_mask_.step, _cudaStreamImage, _cudaStreamMask);
	else
		feedCUDA((uchar*)_img.data, (uchar*)_mask.data, (uchar*)dst_.data, (uchar*)dst_mask_.data, dx, dy, _img.cols, _img.rows, _img.step, dst_.step, _mask.step, dst_mask_.step);
	
}


void CUDABlender::blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj)
{
#ifdef NO_COMPILE
	if (_cudaStreamImage && _cudaStreamMask){
		cudaStreamSynchronize(_cudaStreamImage);
		cudaStreamSynchronize(_cudaStreamMask);
	}
#endif
	cv::cuda::GpuMat mask;
	cv::cuda::compare(dst_mask_, 0, mask, cv::CMP_EQ, streamObj);
	dst_.setTo(cv::Scalar::all(0), mask);
	dst_.copyTo(dst);
	dst_mask_.copyTo(dst_mask);

	dst_.release();
	dst_mask_.release();

}


void CUDABlender::createWeightMap(const cv::cuda::GpuMat& mask, cv::cuda::GpuMat& weight)
{
      cv::Mat _mask, _weight;
      mask.download(_mask);
      cv::distanceTransform(_mask, _weight, cv::DIST_L1, 3);
      weight.upload(_weight);
      cv::cuda::GpuMat temp;
      cv::cuda::multiply(weight, sharpness_, temp);
      cv::cuda::threshold(temp, weight, 1.f, 1.f, cv::THRESH_TRUNC);

}


void CUDABlender::feed_weight(cv::cuda::GpuMat& _img, cv::cuda::GpuMat& _mask, const cv::Point& tl)
{

	CV_Assert(_img.type() == CV_16SC3);
	CV_Assert(_mask.type() == CV_8U);
	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;

	createWeightMap(_mask, weight_map_);

	weightBlendCUDA(_img, weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy);
}





 void CUDABlender::blend_map(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj)
 {
     normalizeUsingWeightMapGpu32F(dst_weight_map_, dst_, dst_weight_map_.cols, dst_weight_map_.rows);
     cv::cuda::compare(dst_weight_map_, WEIGHT_EPS, dst_mask_, cv::CMP_GT);

     cv::cuda::GpuMat mask;
     cv::cuda::compare(dst_mask_, 0, mask, cv::CMP_EQ, streamObj);
     dst_.setTo(cv::Scalar::all(0), mask);
     dst_.copyTo(dst);
     dst_mask_.copyTo(dst_mask);

     dst_.release();
     dst_mask_.release();
 }
