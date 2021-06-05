#include "SVGainCompensator.hpp"
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


SVGainCompensator::SVGainCompensator(const size_t num_imgs_) : imgs_num(num_imgs_)
{
    warp = std::move(std::vector<cv::UMat>(imgs_num));
    mask = std::move(std::vector<cv::UMat>(imgs_num));
}


void SVGainCompensator::computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                                     const std::vector<cv::cuda::GpuMat>& warp_masks)
{
    cv::Ptr<cv::detail::ExposureCompensator> compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN);
    for (auto i = 0; i < imgs_num; ++i){
        warp_imgs[i].download(warp[i]);
        warp_masks[i].download(mask[i]);
    }

    compens->feed(corners, warp, mask);

    std::vector<cv::Mat> gains_;
    compens->getMatGains(gains_);

    gains = cv::Mat_<double>(gains_.size(), 1);

    for (auto i = 0; i < gains_.size(); i++){
       gains(i, 0) = gains_[i].at<double>(0, 0);
    }
}

#include <iostream>
bool SVGainCompensator::apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj)
{
   if (idx > imgs_num || imgs_num <= 0)
     return false;


   cv::cuda::GpuMat temp;
   temp.upload(gains(idx));
   std::cerr << temp.size() << "\n";
   cv::cuda::multiply(warp_img, temp, warp_img, 1, -1);

   return true;
}
