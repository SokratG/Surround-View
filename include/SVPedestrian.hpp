#pragma once
#include <cmath>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>



class SVPedestrianDetection
{
private:
   cv::Ptr<cv::cuda::HOG> hogdetect;
public:
    SVPedestrianDetection();
};
