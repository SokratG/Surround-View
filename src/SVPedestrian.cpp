#include <SVPedestrian.hpp>
#include <opencv2/cudaimgproc.hpp>



SVPedestrianDetection::SVPedestrianDetection(const cv::Size& win_size, const cv::Size& bl_stride, const int nlevels, const double scale,
                                             const double hitThreshold, const int groupThreshold)
{
    hogdetect = cv::cuda::HOG::create(win_size, cv::Size(16, 16), bl_stride);
    hogdetect->setSVMDetector(hogdetect->getDefaultPeopleDetector());

    hogdetect->setNumLevels(nlevels);
    hogdetect->setScaleFactor(scale);
    hogdetect->setHitThreshold(hitThreshold);
    hogdetect->setGroupThreshold(groupThreshold);
}



bool SVPedestrianDetection::detect(cv::cuda::GpuMat& img, std::vector<cv::Rect>& detect_rects)
{
    if (img.empty())
        return false;

    detect_rects.clear();

    cv::cuda::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY, 0, streamObj);

    hogdetect->detectMultiScale(gray_img, detect_rects);

    if (detect_rects.empty())
        return false;

    return true;
}
