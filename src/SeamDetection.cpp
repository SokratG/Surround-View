#include <SeamDetection.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <iostream>

#define EXPERIMENTAL_TEST

bool SeamDetector::init(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& Ks_f, const std::vector<cv::Mat>& R, const std::vector<cv::Mat>& T)
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



bool SeamDetector::warpedImage(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& Ks_f,
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
    //cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::CylindricalWarper>();

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

      std::vector<cv::UMat> wimg(2);
      imgs_warped_f[idxmax].copyTo(wimg[0]); imgs_warped_f[idxmin].copyTo(wimg[1]);
      std::vector<cv::UMat> mimg(2);
      masks_warped_[idxmax].copyTo(mimg[0]); masks_warped_[idxmin].copyTo(mimg[1]);
      std::vector<cv::Point> cor{corners[idxmax], corners[idxmin]};


      /**
        Algorithm:
        1. Threshold images(use mask)
        2. Find contours both thresh. images
        3. Find tl, bl -> first image;  tr, br -> last image
        4. Warp perspective to rectangle or trapezoid both images and masks
        5. Seam find between this warped images
        6. Unwarp mask to previous points
        7. Bitwise-and mask with mask with another seam only for last mask
      */
      /* for last image */
      std::vector<std::vector<cv::Point>> cnts;
      cv::findContours(mimg[0], cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
      cv::Point br(0, 0), tr(0, 0);
      auto idx_tr = 0;
      const auto half_w = mimg[0].cols >> 1;
      for(auto i = 0; i < cnts[0].size(); ++i){
          const auto& pt = cnts[0][i];
          if (br.x < pt.x){
            br = pt;
            idx_tr = i;
          }
      }

      for(auto i = idx_tr; i < cnts[0].size() - 1; ++i){
          const auto& pt = cnts[0][i];
          const auto& next_pt = cnts[0][i + 1];
          auto dy = next_pt.y - pt.y;
          if (pt.x > half_w && br.y > pt.y && dy == 0){
            tr = pt;
            break;
          }
      }

      std::vector<cv::Point_<float>> src_pts{cv::Point(0, 0), cv::Point(0, mimg[0].rows), tr, br};
      std::vector<cv::Point_<float>> def_pts{cv::Point(0, 0), cv::Point(0, mimg[0].rows),
                                             cv::Point(mimg[0].cols, 0), cv::Point(mimg[0].cols, mimg[0].rows)};
      auto transformM = cv::getPerspectiveTransform(src_pts, def_pts);
      cv::warpPerspective(wimg[0], wimg[0], transformM, wimg[0].size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      cv::warpPerspective(mimg[0], mimg[0], transformM, mimg[0].size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);



      /* for first image */
      cv::findContours(mimg[1], cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
      cv::Point bl(0, 0), tl(0, 0);
      auto idx_tl = 0;
      for(auto i = 0; i < cnts[0].size(); ++i){
          const auto& pt = cnts[0][i];
          if (bl.x >= pt.x){
            bl = pt;
            idx_tl = i;
          }
      }
      for(auto i = idx_tl; i >= 0; --i){
          const auto& pt = cnts[0][i + 1];
          const auto& prev_pt = cnts[0][i];
          auto dy = pt.y - prev_pt.y;
          auto dx = pt.x - prev_pt.x;
          if (pt.x < half_w && bl.y > pt.y && dx == 0){
            tl = pt;
            //break;
          }
      }      



      src_pts = std::vector<cv::Point_<float>>{tl, bl, cv::Point(mimg[1].cols, 0), cv::Point(mimg[1].cols, mimg[1].rows)};
      def_pts =  std::vector<cv::Point_<float>>{cv::Point(0, 0), cv::Point(0, mimg[1].rows),
                                                cv::Point(mimg[1].cols, 0), cv::Point(mimg[1].cols, mimg[1].rows)};
      transformM = cv::getPerspectiveTransform(src_pts, def_pts);
      auto inv_transformM = cv::getPerspectiveTransform(def_pts, src_pts);
      cv::warpPerspective(wimg[1], wimg[1], transformM, wimg[1].size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
      cv::warpPerspective(mimg[1], mimg[1], transformM, mimg[1].size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

      constexpr auto scale_factor = 1;
      /* compute new overlap ROI */
      cor[0].x = 0; cor[1].x = sizes[idxmax].width - 200;
      //cv::abs(corners[idxmin].x) - (sizes[idxmax].width / scale_factor);
      cor[0].y = -10; cor[1].y = 10;
      seam_finder->find(wimg, cor, mimg);

      cv::warpPerspective(mimg[1], mimg[1], inv_transformM, mimg[1].size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

      cv::bitwise_and(masks_warped_[idxmin], mimg[1], masks_warped_[idxmin]);

}



