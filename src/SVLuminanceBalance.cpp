#include <SVLuminanceBalance.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

SVLuminanceBalance::SVLuminanceBalance(const size_t imgs_num_, const cv::Size& img_size) : imgs_num(imgs_num_)
{
    chanHSV = std::move(std::vector<std::vector<cv::cuda::GpuMat>>(imgs_num));
    vecHSV = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
    cacheValue = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
    cacheHSV = std::move(std::vector<std::vector<cv::cuda::GpuMat>>(imgs_num));
    cacheVecHSV = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
    ColorValueMean = std::move(cv::cuda::GpuMat(img_size, CV_8UC3));
}

bool SVLuminanceBalance::applyBalance(const int idx, cv::cuda::GpuMat& rgb_img, cv::cuda::Stream& streamObj)
{
    if (idx > imgs_num || imgs_num <= 0)
      return false;

    cv::cuda::cvtColor(rgb_img, vecHSV[idx], cv::COLOR_RGB2HSV, 0, streamObj);
    cv::cuda::split(vecHSV[idx], chanHSV[idx], streamObj);
    // take V component
    cv::cuda::subtract(ColorValueMean, chanHSV[idx][2], cacheValue[idx], cv::noArray(), -1, streamObj);
    cv::cuda::add(chanHSV[idx][2], cacheValue[idx], chanHSV[idx][2], cv::noArray(), -1, streamObj);

    cv::cuda::merge(chanHSV[idx], vecHSV[idx], streamObj);
    cv::cuda::cvtColor(vecHSV[idx], rgb_img, cv::COLOR_HSV2RGB, 0, streamObj);

    return true;
}


void SVLuminanceBalance::compute_MeanLuminance(const std::vector<cv::cuda::GpuMat>& rgb_imgs, cv::cuda::Stream& streamObj)
{

    ColorValueMean.setTo(cv::Scalar(0), streamObj);
    for (auto i = 0; i < imgs_num; ++i){
        cv::cuda::cvtColor(rgb_imgs[i], cacheVecHSV[i], cv::COLOR_RGB2HSV, 0, streamObj);
        cv::cuda::split(cacheVecHSV[i], cacheHSV[i], streamObj);
        // take V component
        cv::cuda::add(ColorValueMean, cacheHSV[i][2], ColorValueMean, cv::noArray(), -1, streamObj);
    }
    cv::cuda::multiply(ColorValueMean, cv::Scalar(1.0/imgs_num), ColorValueMean, 1, -1, streamObj);
}
