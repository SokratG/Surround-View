#include <SVBlender.hpp>

#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>


#include <omp.h>


typedef unsigned char uchar;

extern "C" {
	void feedCUDA(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step);
	void feedCUDA_Async(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask,
			    int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step,
			    cudaStream_t streamimage, cudaStream_t streammask);

	void weightBlendCUDA(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
	    cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, const cv::Size& img_size, int dx, int dy);
	void weightBlendCUDA_Async(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
	    cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, const cv::Size& img_size, int dx, int dy,
				   cudaStream_t stream_dst);

	void addSrcWeightGpu32F(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
				cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, cv::Rect &rc);
	void addSrcWeightGpu32F_Async(const cv::cuda::PtrStep<short> src, const cv::cuda::PtrStepf src_weight,
				cv::cuda::PtrStep<short> dst, cv::cuda::PtrStepf dst_weight, cv::Rect &rc,
				cudaStream_t stream_dst);

	void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
						      const int width, const int height);

	void normalizeUsingWeightMapGpu32F_Async(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<short> src,
						      const int width, const int height, cudaStream_t stream_src);
}

static constexpr float WEIGHT_EPS = 1e-5f;

// ------------------------------- CUDABlender --------------------------------
SVBlender::SVBlender()
{
	if (cudaStreamCreate(&_cudaStreamImage) != cudaError::cudaSuccess)
		_cudaStreamImage = NULL;
	if (cudaStreamCreate(&_cudaStreamMask) != cudaError::cudaSuccess)
		_cudaStreamMask = NULL;
}


SVBlender::~SVBlender(void)
{
	if(_cudaStreamImage)
	   cudaStreamDestroy(_cudaStreamImage);
	if(_cudaStreamMask)
	   cudaStreamDestroy(_cudaStreamMask);
}


void SVBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes)
{
	prepare(cv::detail::resultRoi(corners, sizes));
}


void SVBlender::prepare(cv::Rect dst_roi)
{
	dst_ = cv::cuda::GpuMat(dst_roi.size(), CV_16SC3);
	dst_.setTo(cv::Scalar::all(0));
	dst_mask_ = cv::cuda::GpuMat(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));
	dst_roi_ = dst_roi;
}


void SVBlender::feed(cv::cuda::GpuMat& _img, cv::cuda::GpuMat& _mask, cv::Point tl)
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


void SVBlender::blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj)
{
#ifdef NO_COMPILE
	if (_cudaStreamImage && _cudaStreamMask){
		cudaStreamSynchronize(_cudaStreamImage);
		cudaStreamSynchronize(_cudaStreamMask);
	}
#endif

	cv::cuda::compare(dst_mask_, 0, inter_mask, cv::CMP_EQ, streamObj);
	dst_.setTo(cv::Scalar::all(0), inter_mask, streamObj);
	dst_.convertTo(dst, CV_8U, streamObj);
	dst_mask_.copyTo(dst_mask, streamObj);

	dst_.setTo(cv::Scalar::all(0), cv::noArray(), streamObj);
	dst_mask_.setTo(cv::Scalar::all(0), cv::noArray(), streamObj);
}




// ------------------------------- CUDAFeatherBlender --------------------------------
SVFeatherBlender::SVFeatherBlender(const float sharpness) :
      sharpness_(sharpness), use_cache_weight_(false)
{

    if (cudaStreamCreate(&_cudaStreamDst) != cudaError::cudaSuccess)
            _cudaStreamDst = NULL;
    if (cudaStreamCreate(&_cudaStreamDst_weight) != cudaError::cudaSuccess)
            _cudaStreamDst_weight = NULL;
}

SVFeatherBlender::~SVFeatherBlender()
{
    if(_cudaStreamDst)
       cudaStreamDestroy(_cudaStreamDst);
    if(_cudaStreamDst_weight)
       cudaStreamDestroy(_cudaStreamDst_weight);
}


void SVFeatherBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes)
{
        prepare(cv::detail::resultRoi(corners, sizes));
}


void SVFeatherBlender::prepare(cv::Rect dst_roi)
{
	dst_ = cv::cuda::GpuMat(dst_roi.size(), CV_16SC3);
	dst_.setTo(cv::Scalar::all(0));
	dst_mask_ = cv::cuda::GpuMat(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));
	dst_roi_ = dst_roi;

	dst_weight_map_.create(dst_roi.size(), CV_32F);
	dst_weight_map_.setTo(cv::Scalar::all(0));
}


void SVFeatherBlender::createWeightMap(const cv::cuda::GpuMat& mask, cv::cuda::GpuMat& weight, cv::cuda::Stream& streamObj)
{
      cv::Mat _mask, _weight;
      mask.download(_mask);
      cv::distanceTransform(_mask, _weight, cv::DIST_L1, 3);
      weight.upload(_weight);
      cv::cuda::GpuMat temp;
      cv::cuda::multiply(weight, sharpness_, temp, 1, -1, streamObj);
      cv::cuda::threshold(temp, weight, 1.f, 1.f, cv::THRESH_TRUNC, streamObj);
}

void SVFeatherBlender::feed(cv::cuda::GpuMat& _img, cv::cuda::GpuMat& _mask, const cv::Point& tl, cv::cuda::Stream& streamObj)
{

	CV_Assert(_img.type() == CV_16SC3);
	CV_Assert(_mask.type() == CV_8U);
	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;

	std::unique_ptr<cv::cuda::GpuMat> weight_map_ = std::make_unique<cv::cuda::GpuMat>();
	createWeightMap(_mask, *weight_map_, streamObj);

	if (_cudaStreamDst && _cudaStreamDst_weight)
	    weightBlendCUDA_Async(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy, _cudaStreamDst);
	else
	    weightBlendCUDA(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy);

}

void SVFeatherBlender::feed(cv::cuda::GpuMat& _img, cv::cuda::GpuMat& _mask, const cv::Point& tl, const int idx, cv::cuda::Stream& streamObj)
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
	    weightBlendCUDA_Async(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy, _cudaStreamDst);
	else
	    weightBlendCUDA(_img, *weight_map_, dst_, dst_weight_map_, _img.size(), dx, dy);

}


 void SVFeatherBlender::blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj)
 {

     normalizeUsingWeightMapGpu32F(dst_weight_map_, dst_, dst_weight_map_.cols, dst_weight_map_.rows);

     cv::cuda::compare(dst_weight_map_, WEIGHT_EPS, dst_mask_, cv::CMP_GT, streamObj);


     cv::cuda::compare(dst_mask_, 0, inter_mask, cv::CMP_EQ, streamObj);
     dst_.setTo(cv::Scalar::all(0), inter_mask, streamObj);
     dst_.convertTo(dst, CV_8U, streamObj);
     dst_mask_.copyTo(dst_mask, streamObj);

     dst_.setTo(cv::Scalar::all(0), cv::noArray(), streamObj);
     dst_mask_.setTo(cv::Scalar::all(0), cv::noArray(), streamObj);
     dst_weight_map_.setTo(cv::Scalar::all(0), cv::noArray(), streamObj);
 }


void SVFeatherBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes, const std::vector<cv::cuda::GpuMat>& masks)
{
    prepare(cv::detail::resultRoi(corners, sizes));
    weight_maps_ = std::move(std::vector<cv::cuda::GpuMat>(sizes.size()));

    for (auto i = 0; i < masks.size(); ++i)
        createWeightMap(masks[i], weight_maps_[i]);

    use_cache_weight_ = true;
}






// ------------------------------- CUDAMultiBandBlender --------------------------------
SVMultiBandBlender::SVMultiBandBlender(const int numbands_) : numbands(numbands_)
{
      CV_Assert(numbands_ >= 1);

      if (cudaStreamCreate(&_cudaStreamDst) != cudaError::cudaSuccess)
              _cudaStreamDst = NULL;
}

SVMultiBandBlender::~SVMultiBandBlender()
{
      if(_cudaStreamDst)
         cudaStreamDestroy(_cudaStreamDst);
}


void SVMultiBandBlender::prepare_roi(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes)
{
	prepare_pyr(cv::detail::resultRoi(corners, sizes));

	for (auto i = 0; i < sizes.size(); ++i){
	    const auto& tl =  corners[i];
	    const auto& size_ = sizes[i];
	     // Keep source image in memory with small border
	    int gap = 3 * (1 << numbands);
	    cv::Point tl_new(std::max(dst_roi_.x, tl.x - gap),
			 std::max(dst_roi_.y, tl.y - gap));
	    cv::Point br_new(std::min(dst_roi_.br().x, tl.x + size_.width + gap),
			 std::min(dst_roi_.br().y, tl.y + size_.height + gap));

	    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
	    // After that scale between layers is exactly 2.
	    //
	    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
	    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
	    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> numbands) << numbands);
	    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> numbands) << numbands);
	    auto width = br_new.x - tl_new.x;
	    auto height = br_new.y - tl_new.y;
	    width += ((1 << numbands) - width % (1 << numbands)) % (1 << numbands);
	    height += ((1 << numbands) - height % (1 << numbands)) % (1 << numbands);
	    br_new.x = tl_new.x + width;
	    br_new.y = tl_new.y + height;
	    auto dy = std::max(br_new.y - dst_roi_.br().y, 0);
	    auto dx = std::max(br_new.x - dst_roi_.br().x, 0);
	    tl_new.x -= dx; br_new.x -= dx;
	    tl_new.y -= dy; br_new.y -= dy;

	    auto top = tl.y - tl_new.y;
	    auto left = tl.x - tl_new.x;
	    auto bottom = br_new.y - tl.y - size_.height;
	    auto right = br_new.x - tl.x - size_.width;
	    gpu_imgs_borders_.emplace_back(top, left, bottom, right);
	    gpu_imgs_corners_.emplace_back(tl_new, br_new);

	    gpu_weight_pyr_gauss_vec_.push_back(std::vector<cv::cuda::GpuMat>(numbands + 1));
	    gpu_src_pyr_laplace_vec_.push_back(std::vector<cv::cuda::GpuMat>(numbands + 1));
	    gpu_ups_.push_back(std::vector<cv::cuda::GpuMat>(numbands));
	}

	for (auto i = 0; i < sizes.size(); ++i){
	    gpu_ups_.push_back(std::vector<cv::cuda::GpuMat>(numbands + 1));
	}

}

void SVMultiBandBlender::prepare_pyr(const cv::Rect& dst_roi)
{
	dst_roi_final_ = dst_roi;
	dst_rc_ = cv::Rect(0, 0, dst_roi_final_.width, dst_roi_final_.height);
	dst_roi_ = dst_roi;
	dst_roi_.width += ((1 << numbands) - dst_roi.width % (1 << numbands)) % (1 << numbands);
	dst_roi_.height += ((1 << numbands) - dst_roi.height % (1 << numbands)) % (1 << numbands);
	dst_mask_ = cv::cuda::GpuMat(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));


	gpu_dst_pyr_laplace_.resize(numbands + 1);
	gpu_dst_pyr_laplace_[0].create(dst_roi_.size(), CV_16SC3);
	gpu_dst_pyr_laplace_[0].setTo(cv::Scalar::all(0));

	gpu_dst_band_weights_.resize(numbands + 1);
	gpu_dst_band_weights_[0].create(dst_roi_.size(), CV_32F);
	gpu_dst_band_weights_[0].setTo(0);

	for(auto i = 1; i <= numbands; ++i){
	    auto l_half_rows_ = (gpu_dst_pyr_laplace_[i-1].rows + 1) / 2;
	    auto l_half_cols_ = (gpu_dst_pyr_laplace_[i-1].cols + 1) / 2;
	    gpu_dst_pyr_laplace_[i].create(l_half_rows_, l_half_cols_, CV_16SC3);
	    gpu_dst_pyr_laplace_[i].setTo(cv::Scalar::all(0));

	    auto b_half_rows_ = (gpu_dst_band_weights_[i-1].rows + 1) / 2;
	    auto b_half_cols_ = (gpu_dst_band_weights_[i-1].cols + 1) / 2;
	    gpu_dst_band_weights_[i].create(b_half_rows_, b_half_cols_, CV_32F);
	    gpu_dst_band_weights_[i].setTo(0);
	}

}


void SVMultiBandBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes, const std::vector<cv::cuda::GpuMat>& masks)
{
      prepare_roi(corners, sizes);

      constexpr auto weight_coef = 1. / 255.;

      for(auto i = 0; i < masks.size(); ++i){
          cv::cuda::GpuMat gpu_weight_map_;
          masks[i].convertTo(gpu_weight_map_, CV_32F, weight_coef);
          auto top = gpu_imgs_borders_[i].top;
          auto left = gpu_imgs_borders_[i].left;
          auto bottom = gpu_imgs_borders_[i].bottom;
          auto right = gpu_imgs_borders_[i].right;
          cv::cuda::copyMakeBorder(gpu_weight_map_, gpu_weight_pyr_gauss_vec_[i][0], top, bottom, left, right, cv::BORDER_CONSTANT);
          for (auto j = 0; j < numbands; ++j)
              cv::cuda::pyrDown(gpu_weight_pyr_gauss_vec_[i][j], gpu_weight_pyr_gauss_vec_[i][j + 1]);
      }
}


void SVMultiBandBlender::feed(const cv::cuda::GpuMat& _img, const cv::cuda::GpuMat& _mask, const int idx, cv::cuda::Stream& streamObj)
{
     CV_Assert(_mask.type() == CV_8U);
     feed(_img, idx, streamObj);
}


void SVMultiBandBlender::feed(const cv::cuda::GpuMat& _img, const int idx, cv::cuda::Stream& streamObj)
{
      CV_Assert(_img.type() == CV_16SC3);
      CV_Assert(idx >= 0);

      cv::cuda::copyMakeBorder(_img, gpu_src_pyr_laplace_vec_[idx][0], gpu_imgs_borders_[idx].top, gpu_imgs_borders_[idx].bottom,
                               gpu_imgs_borders_[idx].left, gpu_imgs_borders_[idx].right, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);     

      for(auto i = 0; i < numbands; ++i)
          cv::cuda::pyrDown(gpu_src_pyr_laplace_vec_[idx][i], gpu_src_pyr_laplace_vec_[idx][i + 1], streamObj);

      for(auto i = 0; i < numbands; ++i){
          cv::cuda::pyrUp(gpu_src_pyr_laplace_vec_[idx][i + 1], gpu_ups_[idx][i], streamObj);
          cv::cuda::subtract(gpu_src_pyr_laplace_vec_[idx][i], gpu_ups_[idx][i], gpu_src_pyr_laplace_vec_[idx][i], cv::noArray(), -1, streamObj);
      }

      auto y_tl = gpu_imgs_corners_[idx].tl.y - dst_roi_.y;
      auto y_br = gpu_imgs_corners_[idx].br.y - dst_roi_.y;
      auto x_tl = gpu_imgs_corners_[idx].tl.x - dst_roi_.x;
      auto x_br = gpu_imgs_corners_[idx].br.x - dst_roi_.x;

      for(auto i = 0; i <= numbands; ++i){
           cv::Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);

           auto& src_pyr_laplace = gpu_src_pyr_laplace_vec_[idx][i];

           auto dst_pyr_laplace = gpu_dst_pyr_laplace_[i](rc);

           auto& weight_pyr_gauss = gpu_weight_pyr_gauss_vec_[idx][i];
           auto dst_band_weight = gpu_dst_band_weights_[i](rc);

           addSrcWeightGpu32F_Async(src_pyr_laplace, weight_pyr_gauss, dst_pyr_laplace, dst_band_weight, rc, _cudaStreamDst);

           // div size by 2
           x_tl >>= 1; y_tl >>= 1;
           x_br >>= 1; y_br >>= 1;
      }
}


void SVMultiBandBlender::blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask, cv::cuda::Stream& streamObj)
{
    for (auto i = 0; i <= numbands; ++i){
        auto* dst_i = &gpu_dst_pyr_laplace_[i];
        auto* weight_i = &gpu_dst_band_weights_[i];
        normalizeUsingWeightMapGpu32F_Async(*weight_i, *dst_i, weight_i->cols, weight_i->rows, _cudaStreamDst);
    }

    auto last_idx = gpu_ups_.size() - 1;
    for(size_t i = numbands; i > 0; --i){      
        cv::cuda::pyrUp(gpu_dst_pyr_laplace_[i], gpu_ups_[last_idx][numbands-i], streamObj);
        cv::cuda::add(gpu_ups_[last_idx][numbands-i], gpu_dst_pyr_laplace_[i - 1], gpu_dst_pyr_laplace_[i - 1], cv::noArray(), -1, streamObj);
    }

    cv::cuda::GpuMat mask;
    cv::cuda::compare(gpu_dst_band_weights_[0](dst_rc_), WEIGHT_EPS, dst_mask_, cv::CMP_GT, streamObj);
    cv::cuda::compare(dst_mask_, 0, mask, cv::CMP_EQ, streamObj);

    gpu_dst_pyr_laplace_[0](dst_rc_).setTo(cv::Scalar::all(0), mask, streamObj);
    gpu_dst_pyr_laplace_[0](dst_rc_).convertTo(dst, CV_8U, streamObj);
    dst_mask_.copyTo(dst_mask, streamObj);


#ifndef NO_OMP
    #pragma omp parallel for default(none)
#endif
    for(auto i = 0; i < numbands+1; ++i){
        gpu_dst_band_weights_[i].setTo(0);
        gpu_dst_pyr_laplace_[i].setTo(cv::Scalar::all(0), loopStreamObj);
    }

    dst_mask_.setTo(cv::Scalar::all(0), streamObj);

}



void SVMultiBandBlender::blend(cv::cuda::GpuMat &dst, const bool apply_mask, cv::cuda::Stream& streamObj)
{
    for (auto i = 0; i <= numbands; ++i){
        auto* dst_i = &gpu_dst_pyr_laplace_[i];
        auto* weight_i = &gpu_dst_band_weights_[i];
        normalizeUsingWeightMapGpu32F_Async(*weight_i, *dst_i, weight_i->cols, weight_i->rows, _cudaStreamDst);
    }

    auto last_idx = gpu_ups_.size() - 1;
    for(size_t i = numbands; i > 0; --i){
        cv::cuda::pyrUp(gpu_dst_pyr_laplace_[i], gpu_ups_[last_idx][numbands-i], streamObj);
        cv::cuda::add(gpu_ups_[last_idx][numbands-i], gpu_dst_pyr_laplace_[i - 1], gpu_dst_pyr_laplace_[i - 1], cv::noArray(), -1, streamObj);
    }

    /* this remove some blur around already stitched picture, but if use warp perspective and ROI, we can skip this part */
    if (apply_mask){
        cv::cuda::GpuMat mask;
        cv::cuda::compare(gpu_dst_band_weights_[0](dst_rc_), WEIGHT_EPS, dst_mask_, cv::CMP_GT, streamObj);
        cv::cuda::compare(dst_mask_, 0, mask, cv::CMP_EQ, streamObj);

        gpu_dst_pyr_laplace_[0](dst_rc_).setTo(cv::Scalar::all(0), mask, streamObj);
    }

    gpu_dst_pyr_laplace_[0](dst_rc_).convertTo(dst, CV_8U, streamObj);

#ifndef NO_OMP
  #pragma omp parallel for default(none)
#endif
    for(auto i = 0; i < numbands+1; ++i){
        gpu_dst_band_weights_[i].setTo(0);
        gpu_dst_pyr_laplace_[i].setTo(cv::Scalar::all(0), loopStreamObj);
    }


}
