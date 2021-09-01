#pragma once
#include <vector>

#include <opencv2/core/cuda.hpp>


class SVLuminanceBalance
{
private:
    size_t imgs_num = 0;
    std::vector<std::vector<cv::cuda::GpuMat>> chanHSV, cacheHSV;
    std::vector<cv::cuda::GpuMat> vecHSV, cacheVecHSV, cacheValue;
    cv::cuda::GpuMat ColorValueMean;
public:
    SVLuminanceBalance(const size_t imgs_num_, const cv::Size& img_size);
    bool applyBalance(const int idx, cv::cuda::GpuMat& rgb_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null());
    void compute_MeanLuminance(const std::vector<cv::cuda::GpuMat>& rgb_imgs, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null());
};
