#include <iostream>
#include "AutoCalib.hpp"
#include "SurroundView.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>


bool SurroundView::init(const std::vector<cv::cuda::GpuMat>& imgs){
	
	if (isInit){
	    std::cerr << "SurroundView already initialize...\n";
	    return false;
	}
	
	imgs_num = imgs.size();	

	if (imgs_num <= 1){
	    std::cerr << "Not enough images in imgs vector, must be >= 2...\n";
	    return false;
	}
	
	cv::Size img_size = imgs[0].size();
	
	std::vector<cv::Mat> cpu_imgs(imgs_num);
	for (size_t i = 0; i < imgs_num; ++i){
	    imgs[i].download(cpu_imgs[i]);
	}


	AutoCalib autcalib(imgs_num);
	bool res = autcalib.init(cpu_imgs);
	if (!res){
	    std::cerr << "Error can't autocalibrate camera parameters...\n";
	    return false;
	}
	warped_image_scale = autcalib.get_warpImgScale();
	cameras = autcalib.getExtCameraParam();
	Ks_f = autcalib.getIntCameraParam();


	res = warpImage(cpu_imgs);
	if (!res){
	    std::cerr << "Error can't build warp images...\n";
	    return false;
	}
#ifdef CUT_OFF_FRAME
        res = prepareCutOffFrame(cpu_imgs);
        if (!res){
            std::cerr << "Error can't prepare blending ROI rect...\n";
            return false;
        }
#endif

	cuBlender = std::make_shared<CUDAFeatherBlender>(sharpness);
	cuBlender->prepare(corners, sizes, gpu_seam_masks);

	isInit = true;


	return isInit;
}



bool SurroundView::warpImage(const std::vector<cv::Mat>& imgs)
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


        //cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::PlaneWarper>();
        //cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::SphericalWarper>();
        cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::CompressedRectilinearWarper>(2.f, 1.f);

        cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * work_scale));


        for(size_t i = 0; i < imgs_num; ++i){
              corners[i] = warper->warp(imgs[i], Ks_f[i], cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, imgs_warped[i]);
              sizes[i] = imgs_warped[i].size();
              warper->warp(masks[i], Ks_f[i], cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped_[i]);
              gpu_warpmasks[i].upload(masks_warped_[i]);
        }

	for(auto& msk : masks_warped_){
	      if (msk.cols > mask_maxnorm_size.width || msk.rows > mask_maxnorm_size.height ||
		  msk.cols < mask_minnorm_size.width || msk.rows < mask_minnorm_size.height) {
		      std::cerr << "Error: fail build masks for seam...\n";
		      return false;
	      }
	}


	compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
	compens->feed(corners, imgs_warped, masks_warped_);
	for (int i = 0; i < imgs_num; ++i){
	      compens->apply(i, corners[i], imgs_warped[i], masks_warped_[i]);
	      imgs_warped[i].convertTo(imgs_warped_f[i], CV_32F);
	}


	cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::detail::SeamFinder::createDefault(cv::detail::SeamFinder::VORONOI_SEAM);


	seam_finder->find(imgs_warped_f, corners, masks_warped_);


	cv::Mat morphel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, masks_warped_[0].type(), morphel);

	cv::cuda::GpuMat tempmask, gpu_dilate_mask, gpu_seam_mask;
	cv::Mat xmap, ymap;
	for(size_t i = 0; i < imgs_num; ++i){
		tempmask.upload(masks_warped_[i]);
		dilateFilter->apply(tempmask, gpu_dilate_mask);
		cv::cuda::resize(gpu_dilate_mask, gpu_seam_mask, tempmask.size());
		cv::cuda::bitwise_and(gpu_seam_mask, gpu_warpmasks[i], gpu_seam_masks[i]);
		warper->buildMaps(imgs[i].size(), Ks_f[i], cameras[i].R, xmap, ymap);
		texXmap[i].upload(xmap);
		texYmap[i].upload(ymap);
	}


	return true;
}


#ifdef CUT_OFF_FRAME
bool SurroundView::prepareCutOffFrame(const std::vector<cv::Mat>& cpu_imgs)
{
          cv::detail::MultiBandBlender blender(true, 5);

          blender.prepare(cv::detail::resultRoi(corners, sizes));

          cv::cuda::GpuMat warp_, warp_s, warp_img;
          for(size_t i = 0; i < imgs_num; ++i){
                  warp_.upload(cpu_imgs[i]);
                  cv::cuda::remap(warp_, warp_img, texXmap[i], texYmap[i], cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);
                  warp_img.convertTo(warp_s, CV_16S);
                  blender.feed(warp_s, gpu_seam_masks[i], corners[i]);
          }

          cv::Mat result, mask;
          blender.blend(result, mask);
          result.convertTo(result, CV_8U);

          cv::Mat thresh;
          cv::cvtColor(result, thresh, cv::COLOR_RGB2GRAY);
          cv::threshold(thresh, thresh, 64, 255, cv::THRESH_BINARY);
          cv::Canny(thresh, thresh, 1, 255);

          cv::morphologyEx(thresh, thresh, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
          cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

          /* scan line detect edges in the middle */
          auto middle_x = (result.cols % 2 == 0) ? (result.cols / 2) : (result.cols / 2 + 1);
          const cv::Mat col = result.col(middle_x);
          auto y_top = 0, y_bot = result.rows;

          for (;y_top <= result.rows || y_bot >= 0;){
               auto val_top = col.at<uchar>(y_top);
               auto val_bot = col.at<uchar>(y_bot);
               if (val_top < threshold_color){
                  y_top += 1;
               }
               if (val_bot < threshold_color){
                   y_bot -= 1;
               }
               if (val_bot >= threshold_color && val_top >= threshold_color)
                 break;
          }

          CV_Assert(y_top >= 0 && y_bot <= result.rows);

          //constexpr auto tb_remove = 0.01;
          resSize = result.size();
          blendingEdges = cv::Range(y_top, y_bot);

          return true;
}
#endif




bool SurroundView::stitch(const std::vector<cv::cuda::GpuMat*>& imgs, cv::cuda::GpuMat& blend_img)
{
    if (!isInit){
        std::cerr << "SurroundView was not initialized...\n";
        return false;
    }

    cv::cuda::GpuMat gpuimg_warped_s, gpuimg_warped;
    cv::cuda::GpuMat stitch, mask_, temp;

    #pragma omp parallel for default(none) shared(imgs) private(gpuimg_warped, gpuimg_warped_s)
    for(size_t i = 0; i < imgs_num; ++i){

          cv::cuda::remap(*imgs[i], gpuimg_warped, texXmap[i], texYmap[i], cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);

          gpuimg_warped.convertTo(gpuimg_warped_s, CV_16S, streamObj);

          cuBlender->feed(gpuimg_warped_s, gpu_seam_masks[i], corners[i], i, streamObj);

    }

    cuBlender->blend(stitch, mask_, streamObj);

#ifdef COLOR_CORRECTION

    stitch.convertTo(gpuimg_warped, CV_8U, streamObj);

    cv::cuda::cvtColor(gpuimg_warped, gpuimg_warped, cv::COLOR_RGB2YCrCb, 0, streamObj);

    cv::cuda::split(gpuimg_warped, inrgb, streamObj);

    cv::cuda::equalizeHist(inrgb[0], inrgb[0], streamObj);

    cv::cuda::merge(inrgb, gpuimg_warped, streamObj);

    cv::cuda::cvtColor(gpuimg_warped, blend_img, cv::COLOR_YCrCb2RGB, 0, streamObj);

#else
    //temp = stitch(cv::Range(blendingEdges.start, blendingEdges.end), cv::Range(0, stitch.cols));
    stitch.convertTo(blend_img, CV_8U, streamObj);

#endif
    return true;
}





