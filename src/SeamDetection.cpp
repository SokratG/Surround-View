#include <SeamDetection.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <iostream>



bool SeamDetector::init(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& Ks_f, const std::vector<cv::Mat>& R)
{
        if (isInit){
            std::cerr << "SeamDetector already initialize...\n";
            return isInit;
        }

        bool res = warpedImage(imgs, Ks_f, R);
        if (!res){
            return false;
        }



        isInit = true;

        return isInit;
}


bool SeamDetector::warpedImage(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& Ks_f, const std::vector<cv::Mat>& R)
{
    gpu_seam_masks = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
    corners = std::move(std::vector<cv::Point>(imgs_num));
    sizes = std::move(std::vector<cv::Size>(imgs_num));
    texXmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
    texYmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));

    /* warped images and masks */
    std::vector<cv::UMat> masks_warped_(imgs_num);
    std::vector<cv::UMat> imgs_warped(imgs_num);
    std::vector<cv::UMat> imgs_warped_f(imgs_num);
    std::vector<cv::Mat> masks(imgs_num);
    std::vector<cv::cuda::GpuMat> gpu_warpmasks(imgs_num);


    for (size_t i = 0; i < imgs_num; ++i){
          masks[i].create(imgs[i].size(), CV_8U);
          masks[i].setTo(cv::Scalar::all(255));
    }


    //cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::SphericalWarper>();
    cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::CylindricalWarper>();

    cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * work_scale));

    for(size_t i = 0; i < imgs_num; ++i){
          corners[i] = warper->warp(imgs[i], Ks_f[i], R[i], cv::INTER_LINEAR, cv::BORDER_REFLECT, imgs_warped[i]);
          sizes[i] = imgs_warped[i].size();
          warper->warp(masks[i], Ks_f[i], R[i], cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped_[i]);
          gpu_warpmasks[i].upload(masks_warped_[i]);
    }


    for(const auto& msk : masks_warped_){
          if (msk.cols > mask_maxnorm_size.width || msk.rows > mask_maxnorm_size.height ||
              msk.cols < mask_minnorm_size.width || msk.rows < mask_minnorm_size.height) {
                  std::cerr << msk.size() << "\n";
                  std::cerr << "Error: fail build masks for seam...\n";
                  return false;
          }
    }


    cv::Ptr<cv::detail::ExposureCompensator> compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
    compens->feed(corners, imgs_warped, masks_warped_);
    for (int i = 0; i < imgs_num; ++i){
          compens->apply(i, corners[i], imgs_warped[i], masks_warped_[i]);
          imgs_warped[i].convertTo(imgs_warped_f[i], CV_32F);

    }

    //cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::VORONOI_SEAM);
    cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::DP_SEAM);
    seam_finder->find(imgs_warped_f, corners, masks_warped_);


    cv::Mat morphel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, masks_warped_[0].type(), morphel);

    cv::cuda::GpuMat tempmask, gpu_dilate_mask, gpu_seam_mask;
    cv::Mat xmap, ymap;

    for(size_t i = 0; i < imgs_num; ++i){
            tempmask.upload(masks_warped_[i]);
            dilateFilter->apply(tempmask, gpu_dilate_mask);
            cv::cuda::resize(gpu_dilate_mask, gpu_seam_mask, tempmask.size());
            cv::cuda::bitwise_and(gpu_seam_mask, gpu_warpmasks[i], gpu_seam_masks[i]);
            warper->buildMaps(imgs[i].size(), Ks_f[i], R[i], xmap, ymap);
            texXmap[i].upload(xmap);
            texYmap[i].upload(ymap);
    }

    return true;
}






