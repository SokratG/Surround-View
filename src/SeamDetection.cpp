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


        gpu_seam_masks = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
        corners = std::move(std::vector<cv::Point>(imgs_num));
        sizes = std::move(std::vector<cv::Size>(imgs_num));
        texXmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
        texYmap = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));

        std::vector<cv::UMat> imgs_warped(imgs_num);
        std::vector<cv::UMat> masks_warped_(imgs_num);

        bool res = warpedImage(imgs, Ks_f, R, imgs_warped, masks_warped_);
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


bool SeamDetector::warpedImage(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& Ks_f, const std::vector<cv::Mat>& R,
                 std::vector<cv::UMat>& imgs_warped, std::vector<cv::UMat>& masks_warped_)
{

    /* warped images and masks */
    std::vector<cv::Mat> masks(imgs_num);
    cv::Mat xmap, ymap;

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
          warper->buildMaps(imgs[i].size(), Ks_f[i], R[i], xmap, ymap);
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


bool SeamDetector::seamDetect(const std::vector<cv::UMat>& imgs_warped, std::vector<cv::UMat>& masks_warped_)
{
      std::vector<cv::UMat> imgs_warped_f(imgs_num);
      std::vector<cv::cuda::GpuMat> gpu_warpmasks(imgs_num);

      cv::Ptr<cv::detail::ExposureCompensator> compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
      compens->feed(corners, imgs_warped, masks_warped_);

#ifdef EXPERIMENTAL_TEST
      auto idxmin = 0, idxmax = 0;
      auto max_w = corners[idxmax].x, min_w = corners[idxmin].x;
#endif

      for (int i = 0; i < imgs_num; ++i){
            compens->apply(i, corners[i], imgs_warped[i], masks_warped_[i]);
            imgs_warped[i].convertTo(imgs_warped_f[i], CV_32F);
            gpu_warpmasks[i].upload(masks_warped_[i]);
#ifdef EXPERIMENTAL_TEST
            if (max_w < corners[i].x){
              idxmax = i;
              max_w = corners[i].x;
            }
            if (min_w > corners[i].x){
              idxmin = i;
              min_w = corners[i].x;
            }
#endif
      }


      //cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::VORONOI_SEAM);
      cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::DP_SEAM);
      seam_finder->find(imgs_warped_f, corners, masks_warped_);

#ifdef EXPERIMENTAL_TEST
      fl_seam_detect(imgs_warped_f, masks_warped_, idxmax, idxmin);
#endif

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




void SeamDetector::fl_seam_detect(const std::vector<cv::UMat>& imgs_warped_f, std::vector<cv::UMat>& masks_warped_, int idxmax, int idxmin)
{
      cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::DP_SEAM);

      std::vector<cv::UMat> wimg{imgs_warped_f[idxmax], imgs_warped_f[idxmin]};
      std::vector<cv::UMat> mimg(2);
      masks_warped_[idxmax].copyTo(mimg[0]);
      masks_warped_[idxmin].copyTo(mimg[1]);
      std::vector<cv::Point> cor{corners[idxmax], corners[idxmin]};


      std::vector<cv::Point> last_pts(4);
      std::vector<cv::Point> first_pts(4);

      /**
        Algorithm:
        1. Threshold images
        2. Find contours both thresh. images
        3. Find tl, bl, tr, br points both thresh. images
        4. Warp perspective to rectangle or trapezoid both images and masks
        5. Seam find between this warped images
        6. Unwarp masks to previous points
        7. Bitwise-and mask with mask with another seam only for last mask
      */


      /* compute new overlap ROI */
      cor[1].x = cv::abs(corners[idxmin].x) - cv::abs(cv::abs(corners[idxmax].x) - sizes[idxmax].width);

      seam_finder->find(wimg, cor, mimg);

      cv::bitwise_and(masks_warped_[idxmax], mimg[0], masks_warped_[idxmax]);

      cv::Mat temp, mtemp;
      wimg[0].convertTo(temp, CV_8U);
      cv::bitwise_and(temp, temp, mtemp, mimg[0]);
      // show result...
}



