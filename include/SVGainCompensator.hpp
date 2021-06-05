#pragma once
#include <opencv2/cudaimgproc.hpp>

#include <vector>
#include <cmath>



class SVGainCompensator
{
private:
    size_t imgs_num = 0;
    std::vector<cv::UMat> warp, mask;
private:
    cv::Mat_<double> gains;
    cv::Scalar gain_scalar;
public:
    SVGainCompensator(const size_t num_imgs);
    void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks);
    bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null());
};
