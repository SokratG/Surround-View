#include <SVPedestrian.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <omp.h>

SVPedestrianDetection::SVPedestrianDetection(const size_t imgs_num_, const double img_scale_factor,
                                             const cv::Size& win_size, const cv::Size& bl_stride, const int nlevels,
                                             const double scale, const double hitThreshold, const int groupThreshold) :
                                             scale_factor(img_scale_factor), imgs_num(imgs_num_)
{
    hogdetect = cv::cuda::HOG::create(win_size, cv::Size(16, 16), bl_stride);
    hogdetect->setSVMDetector(hogdetect->getDefaultPeopleDetector());

    hogdetect->setNumLevels(nlevels);
    hogdetect->setScaleFactor(scale);
    hogdetect->setHitThreshold(hitThreshold);
    hogdetect->setGroupThreshold(groupThreshold);

    gpu_warped_scale = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));
    gpu_gray_imgs = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));
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


bool SVPedestrianDetection::detect(std::vector<cv::cuda::GpuMat>& imgs, std::vector<std::vector<cv::Rect>>& detect_rects)
{
    if (imgs.empty())
        return false;

#ifndef NO_OMP
    #pragma omp parallel for default(none) shared(imgs, detect_rects)
#endif
    for(auto i = 1; i <= imgs_num; ++i){
        const auto s = i - 1;
        cv::cuda::resize(imgs[i], gpu_warped_scale[s], cv::Size(), scale_factor, scale_factor, cv::INTER_NEAREST, streamObj);

        cv::cuda::cvtColor(gpu_warped_scale[s], gpu_gray_imgs[s], cv::COLOR_RGB2GRAY, 0, streamObj);

        hogdetect->detectMultiScale(gpu_gray_imgs[s], detect_rects[s]);
    }

    if (detect_rects.empty())
        return false;

    return true;
}
