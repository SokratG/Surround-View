#pragma once
#include <vector>

#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/core/cuda.hpp>


class SVExposureCompensator
{
protected:
    size_t imgs_num = 0;
    std::vector<cv::UMat> warp, mask;
    cv::Ptr<cv::detail::ExposureCompensator> compens;
public:
    SVExposureCompensator(const size_t imgs_num_);
    virtual void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks) = 0;
    virtual bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null()) = 0;
};


class SVGainCompensator : public SVExposureCompensator
{
private:
    cv::Mat_<double> gains;
public:
    SVGainCompensator(const size_t imgs_num_, const int nr_feeds=1);
    void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks) override;
    bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null()) override;
};



class SVGainBlocksCompensator : public SVExposureCompensator
{
private:
    std::vector<cv::cuda::GpuMat> gain;
private:
    std::vector<cv::cuda::GpuMat> gain_map;
    std::vector<cv::cuda::GpuMat> gain_channels;
public:
    SVGainBlocksCompensator(const size_t imgs_num_, const int bl_width=32,
                            const int bl_height=32, const int nr_feeds=1);
    void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks) override;
    bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null()) override;
};



class SVChannelCompensator : public SVExposureCompensator
{
private:
    cv::Mat_<double> gains;
public:
    SVChannelCompensator(const size_t imgs_num_, const int nr_feeds=1);
    void computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                      const std::vector<cv::cuda::GpuMat>& warp_masks) override;
    bool apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj = cv::cuda::Stream::Null()) override;
};
