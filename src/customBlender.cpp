#include "customBlender.h"
#include <opencv2/cudaarithm.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>


typedef unsigned char uchar;

extern "C" {
	void feedCUDA(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step);
	void feedCUDA_Async(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step, cudaStream_t streamimage, cudaStream_t streammask);
	void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
						      const int width, const int height);
	void weightBlendCUDA(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
	    cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, const cv::Size& img_size, int dx, int dy);
	void weightBlendCUDA_Async(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
	    cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, const cv::Size& img_size, int dx, int dy,
				   cudaStream_t stream_dst, cudaStream_t stream_dst_weight);
	void normalizeUsingWeightMapGpu32F_Async(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
						      const int width, const int height, cudaStream_t stream_dst);
}

static constexpr float WEIGHT_EPS = 1e-5f;

// ------------------------------- CUDABlender --------------------------------

CUDABlender::CUDABlender()
{
	if (cudaStreamCreate(&_cudaStreamImage) != cudaError::cudaSuccess)
		_cudaStreamImage = NULL;
	if (cudaStreamCreate(&_cudaStreamMask) != cudaError::cudaSuccess)
		_cudaStreamMask = NULL;
}


CUDABlender::~CUDABlender(void)
{
	if(_cudaStreamImage)
	   cudaStreamDestroy(_cudaStreamImage);
	if(_cudaStreamMask)
	   cudaStreamDestroy(_cudaStreamMask);
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




// ------------------------------- CUDAFeatherBlender --------------------------------

CUDAFeatherBlender::CUDAFeatherBlender(const float sharpness) :
      sharpness_(sharpness), use_cache_weight_(false)
{

    if (cudaStreamCreate(&_cudaStreamDst) != cudaError::cudaSuccess)
            _cudaStreamDst = NULL;
    if (cudaStreamCreate(&_cudaStreamDst_weight) != cudaError::cudaSuccess)
            _cudaStreamDst_weight = NULL;
}

CUDAFeatherBlender::~CUDAFeatherBlender()
{
    if(_cudaStreamDst)
       cudaStreamDestroy(_cudaStreamDst);
    if(_cudaStreamDst_weight)
       cudaStreamDestroy(_cudaStreamDst_weight);
}


void CUDAFeatherBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes)
{
        prepare(cv::detail::resultRoi(corners, sizes));
}


void CUDAFeatherBlender::prepare(cv::Rect dst_roi)
{
	dst_ = cv::cuda::GpuMat(dst_roi.size(), CV_16SC3);
	dst_.setTo(cv::Scalar::all(0));
	dst_mask_ = cv::cuda::GpuMat(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));
	dst_roi_ = dst_roi;

	dst_weight_map_.create(dst_roi.size(), CV_32F);
	dst_weight_map_.setTo(cv::Scalar::all(0));
}


void CUDAFeatherBlender::createWeightMap(const cv::cuda::GpuMat& mask, cv::cuda::GpuMat& weight, cv::cuda::Stream& streamObj)
{
      cv::Mat _mask, _weight;
      mask.download(_mask);
      cv::distanceTransform(_mask, _weight, cv::DIST_L1, 3);
      weight.upload(_weight);
      cv::cuda::GpuMat temp;
      cv::cuda::multiply(weight, sharpness_, temp, 1, -1, streamObj);
      cv::cuda::threshold(temp, weight, 1.f, 1.f, cv::THRESH_TRUNC, streamObj);

}

void CUDAFeatherBlender::feed(cv::cuda::GpuMat& _img, cv::cuda::GpuMat& _mask, const cv::Point& tl, cv::cuda::Stream& streamObj)
{

	CV_Assert(_img.type() == CV_16SC3);
	CV_Assert(_mask.type() == CV_8U);
	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;

	std::unique_ptr<cv::cuda::GpuMat> weight_map_ = std::make_unique<cv::cuda::GpuMat>();
	createWeightMap(_mask, *weight_map_, streamObj);

	if (_cudaStreamDst && _cudaStreamDst_weight)
	    weightBlendCUDA_Async(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy, _cudaStreamDst, _cudaStreamDst_weight);
	else
	    weightBlendCUDA(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy);

}

void CUDAFeatherBlender::feed(cv::cuda::GpuMat& _img, cv::cuda::GpuMat& _mask, const cv::Point& tl, const int idx, cv::cuda::Stream& streamObj)
{

	CV_Assert(_img.type() == CV_16SC3);
	CV_Assert(_mask.type() == CV_8U);
	CV_Assert(idx >= 0);
	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;

	std::unique_ptr<cv::cuda::GpuMat> weight_map_ = std::make_unique<cv::cuda::GpuMat>();
	if (!use_cache_weight_)
	    createWeightMap(_mask, *weight_map_, streamObj);
	else{
	    weight_map_ = std::make_unique<cv::cuda::GpuMat>(weight_maps_[idx]);
	}

	if (_cudaStreamDst && _cudaStreamDst_weight)
	    weightBlendCUDA_Async(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy, _cudaStreamDst, _cudaStreamDst_weight);
	else
	    weightBlendCUDA(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy);

}





 void CUDAFeatherBlender::blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj)
 {
     if (_cudaStreamDst)
        normalizeUsingWeightMapGpu32F(dst_weight_map_, dst_, dst_weight_map_.cols, dst_weight_map_.rows);
     else
        normalizeUsingWeightMapGpu32F_Async(dst_weight_map_, dst_, dst_weight_map_.cols, dst_weight_map_.rows, _cudaStreamDst);

     cv::cuda::compare(dst_weight_map_, WEIGHT_EPS, dst_mask_, cv::CMP_GT, streamObj);

     cv::cuda::GpuMat mask;
     cv::cuda::compare(dst_mask_, 0, mask, cv::CMP_EQ, streamObj);
     dst_.setTo(cv::Scalar::all(0), mask, streamObj);
     dst_.copyTo(dst, streamObj);
     dst_mask_.copyTo(dst_mask, streamObj);

     dst_.setTo(cv::Scalar::all(0));
     dst_mask_.setTo(cv::Scalar::all(0));
     dst_weight_map_.setTo(cv::Scalar::all(0));
 }



void CUDAFeatherBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes, const std::vector<cv::cuda::GpuMat>& masks)
{
    prepare(cv::detail::resultRoi(corners, sizes));
    weight_maps_ = std::move(std::vector<cv::cuda::GpuMat>(sizes.size()));

    for (auto i = 0; i < masks.size(); ++i)
        createWeightMap(masks[i], weight_maps_[i]);

    use_cache_weight_ = true;
}


