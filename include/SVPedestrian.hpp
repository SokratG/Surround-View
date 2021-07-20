#pragma once
#include <cmath>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>

/*
   Implementation based on HOG descriptors is very slow, try use DeepStream SDK
*/

class SVPedestrianDetection
{
private:
   double scale_factor;
   size_t imgs_num;
   std::vector<cv::cuda::GpuMat> gpu_warped_scale;
   std::vector<cv::cuda::GpuMat> gpu_gray_imgs;
   cv::Ptr<cv::cuda::HOG> hogdetect;
   cv::cuda::GpuMat gray_img;
   cv::cuda::Stream streamObj;
public:
    SVPedestrianDetection(const size_t imgs_num_, const double img_scale_factor=1.0,
                          const cv::Size& win_size=cv::Size(64, 128), const cv::Size& bl_stride=cv::Size(8, 8),
                          const int nlevels=64, const double scale=1.05,
                          const double hitThreshold=0.0, const int groupThreshold=0);
    bool detect(cv::cuda::GpuMat& img, std::vector<cv::Rect>& detect_rects);
    bool detect(std::vector<cv::cuda::GpuMat>& imgs, std::vector<std::vector<cv::Rect>>& detect_rects);
};


using SVPedDetect = SVPedestrianDetection;
