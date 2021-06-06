#pragma once
#include <vector>

#include <opencv2/core/cuda.hpp>

class SVGainCompensator
{
private:
    size_t imgs_num = 0;
    std::vector<cv::UMat> warp, mask;
private:
    cv::Mat_<double> gains;
public:
    SVGainCompensator(const size_t imgs_num_);
    void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks);
    bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null());
};



class SVGainBlocksCompensator
{
private:
    size_t imgs_num = 0;
    std::vector<cv::UMat> warp, mask;
    std::vector<cv::cuda::GpuMat> gain;
private:
    std::vector<cv::cuda::GpuMat> gain_map;
    std::vector<cv::cuda::GpuMat> gain_channels;
public:
    SVGainBlocksCompensator(const size_t imgs_num_);
    void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks);
    bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null());
};



class SVChannelCompensator
{
private:
    size_t imgs_num = 0;
    std::vector<cv::UMat> warp, mask;
private:
    cv::Mat_<double> gains;
public:
    SVChannelCompensator(const size_t imgs_num_);
    void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks);
    bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null());
};
