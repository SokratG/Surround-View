#include <SVSeamDetection.hpp>

#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <iostream>


bool SVSeamDetector::find_seam(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& Ks_f, const std::vector<cv::Mat>& R, const std::vector<cv::Mat>& T)
{
        if (isInit){
            std::cerr << "SeamDetector already initialize...\n";
            return isInit;
        }


        gpu_seam_masks = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
        corners = std::move(std::vector<cv::Point>(imgs_num));
        sizes = std::move(std::vector<cv::Size>(imgs_num));
        texXmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
        texYmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));

        std::vector<cv::UMat> imgs_warped(imgs_num);
        std::vector<cv::UMat> masks_warped_(imgs_num);

        bool res = warpedImage(imgs, Ks_f, R, T, imgs_warped, masks_warped_);
        if (!res){
            return false;
        }

        res = seamDetect(imgs_warped, masks_warped_);
        if (!res){
            return false;
        }


        isInit = true;

        return isInit;
}


bool SVSeamDetector::warpedImage(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& Ks_f,
                               const std::vector<cv::Mat>& R, const std::vector<cv::Mat>& T,
                                std::vector<cv::UMat>& imgs_warped, std::vector<cv::UMat>& masks_warped_)
{

    /* warped images and masks */
    std::vector<cv::Mat> masks(imgs_num);
    cv::Mat xmap, ymap;

    for (size_t i = 0; i < imgs_num; ++i){
          masks[i].create(imgs[i].size(), CV_8U);
          masks[i].setTo(cv::Scalar::all(255));
    }


    cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::SphericalWarper>();

    cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(work_scale * warped_image_scale));

    cv::Mat_<float> K_;
    for(size_t i = 0; i < imgs_num; ++i){
          Ks_f[i].copyTo(K_);
          K_(0,0) *= work_scale; K_(0,2) *= work_scale;
          K_(1,1) *= work_scale; K_(1,2) *= work_scale;
          corners[i] = warper->warp(imgs[i], K_, R[i], cv::INTER_LINEAR, cv::BORDER_REFLECT, imgs_warped[i]);
          sizes[i] = imgs_warped[i].size();
          warper->warp(masks[i], K_, R[i], cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped_[i]);
          warper->buildMaps(imgs[i].size(), K_, R[i], xmap, ymap);
          texXmap[i].upload(xmap);
          texYmap[i].upload(ymap);
    }


    for(const auto& msk : masks_warped_){
          if (msk.cols > mask_maxnorm_size.width || msk.rows > mask_maxnorm_size.height ||
              msk.cols < mask_minnorm_size.width || msk.rows < mask_minnorm_size.height) {
                  std::cerr << msk.size() << "\n";
                  std::cerr << "Error: fail build masks for seam...\n";
                  return false;
          }
    }


    return true;
}


bool SVSeamDetector::seamDetect(const std::vector<cv::UMat>& imgs_warped, std::vector<cv::UMat>& masks_warped_)
{
      std::vector<cv::UMat> imgs_warped_f(imgs_num);
      std::vector<cv::cuda::GpuMat> gpu_warpmasks(imgs_num);

      cv::Ptr<cv::detail::ExposureCompensator> compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
      compens->feed(corners, imgs_warped, masks_warped_);


      for (int i = 0; i < imgs_num; ++i){
            compens->apply(i, corners[i], imgs_warped[i], masks_warped_[i]);
            imgs_warped[i].convertTo(imgs_warped_f[i], CV_32F);
            gpu_warpmasks[i].upload(masks_warped_[i]);
      }

      //cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::VORONOI_SEAM);
      cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::DP_SEAM);
      seam_finder->find(imgs_warped_f, corners, masks_warped_);


      cv::Mat morphel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
      cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, masks_warped_[0].type(), morphel);

      cv::cuda::GpuMat tempmask, gpu_dilate_mask, gpu_seam_mask;


      for(size_t i = 0; i < imgs_num; ++i){
              tempmask.upload(masks_warped_[i]);
              dilateFilter->apply(tempmask, gpu_dilate_mask);
              cv::cuda::resize(gpu_dilate_mask, gpu_seam_mask, tempmask.size());
              cv::cuda::bitwise_and(gpu_seam_mask, gpu_warpmasks[i], gpu_seam_masks[i]);
      }

      return true;
}

