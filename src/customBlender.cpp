#include "customBlender.h"
#include <opencv2/cudaarithm.hpp>
#include "opencv2/highgui.hpp"


typedef unsigned char uchar;

extern "C" {
	void feedCUDA(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step);
	void feedCUDA_Async(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step, cudaStream_t streamimage, cudaStream_t streammask);
}

CustomBlender::CustomBlender(void)
{
	if (cudaStreamCreate(&_cudaStreamImage) != cudaError::cudaSuccess)
		_cudaStreamImage = NULL;
	if (cudaStreamCreate(&_cudaStreamMask) != cudaError::cudaSuccess)
		_cudaStreamMask = NULL;
}


CustomBlender::~CustomBlender(void)
{
	
}


void CustomBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes)
{
	prepare(cv::detail::resultRoi(corners, sizes));
}


void CustomBlender::prepare(cv::Rect dst_roi)
{
	dst_ = cv::cuda::GpuMat(dst_roi.size(), CV_16SC3);
	dst_.setTo(cv::Scalar::all(0));
	dst_mask_ = cv::cuda::GpuMat(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));
	dst_roi_ = dst_roi;
}


void CustomBlender::feed(cv::cuda::GpuMat _img, cv::cuda::GpuMat _mask, cv::Point tl)
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


void CustomBlender::blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask)
{
	if (_cudaStreamImage && _cudaStreamMask){
		cudaStreamSynchronize(_cudaStreamImage);
		cudaStreamSynchronize(_cudaStreamMask);
	}

	cv::cuda::GpuMat mask;
	cv::cuda::compare(dst_mask_, 0, mask, cv::CMP_EQ);
	dst_.setTo(cv::Scalar::all(0), mask);
	dst_.copyTo(dst);
	dst_mask_.copyTo(dst_mask);

	dst_.release();
	dst_mask_.release();

}
