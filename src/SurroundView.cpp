#include <iostream>
#include "AutoCalib.hpp"
#include "SurroundView.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

//#define LOG



bool SurroundView::init(const std::vector<cv::cuda::GpuMat>& imgs, const std::vector<cv::Mat>& intrisicMat){
	
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
	
	std::vector<cv::Mat> rescale_imgs(imgs_num);
	for (size_t i = 0; i < imgs_num; ++i){
	    if (!work_set){
		    work_scale = std::min(1.0, std::sqrt(registr_resol * 1e6 / imgs[i].size().area()));
		    work_set = true;
	    }
	    imgs[i].download(rescale_imgs[i]);
	    //cv::resize(rescale_imgs[i], rescale_imgs[i], cv::Size(), work_scale, work_scale, cv::INTER_AREA);
	}


	AutoCalib autcalib(imgs_num);
	bool res = autcalib.init(rescale_imgs, intrisicMat);
	if (!res){
	    std::cerr << "Error can't autocalibrate camera parameters...\n";
	    return false;
	}
	warped_image_scale = autcalib.get_warpImgScale();
	cameras = autcalib.getExtCameraParam();
	Ks_f = autcalib.getIntCameraParam();


	res = warpImage(rescale_imgs);
	if (!res){
	    std::cerr << "Error can't build warp images...\n";
	    return false;
	}


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



        cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::AffineWarper>();
        //cv::Ptr<cv::WarperCreator> warper_creator = cv::makePtr<cv::CompressedRectilinearWarper>(2.f, 1.f);

        cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * work_scale));


        for(size_t i = 0; i < imgs_num; ++i){
              corners[i] = warper->warp(imgs[i], Ks_f[i], cameras[i].R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, imgs_warped[i]);
              sizes[i] = imgs_warped[i].size();
              warper->warp(masks[i], Ks_f[i], cameras[i].R, cv::INTER_NEAREST, cv::BORDER_REFLECT, masks_warped_[i]);
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
		warper->buildMaps(sizes[i], Ks_f[i], cameras[i].R, xmap, ymap);
		texXmap[i].upload(xmap);
		texYmap[i].upload(ymap);
	}


	if (!prepareGainMatrices(imgs_warped)){
	    std::cerr << "Error: fail build gain compensator matrices...\n";
	    return false;
	}


	return true;
}



bool SurroundView::prepareGainMatrices(const std::vector<cv::UMat>& warp_imgs)
{
	std::vector<cv::Mat> gain_map;

	compens->getMatGains(gain_map);

	if (gain_map.size() == 0){
		std::cerr << "Error: no gain matrices for exposure compensator...\n";
		return false;
	}
	if (gain_map.size() != imgs_num){
		std::cerr << "Error: wrong size gain matrices for exposure compensator...\n";
		return false;
	}

	gpu_gain_map = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));

	for (size_t i = 0; i < imgs_num; ++i){
	      cv::resize(gain_map[i], gain_map[i], warp_imgs[i].size(), 0., 0., cv::INTER_LINEAR);

	      if (gain_map[i].channels() != 3){
		    std::vector<cv::Mat> gains_channels;
		    gains_channels.push_back(gain_map[i]);
		    gains_channels.push_back(gain_map[i]);
		    gains_channels.push_back(gain_map[i]);
		    cv::merge(gains_channels, gain_map[i]);
	      }
	      gpu_gain_map[i].upload(gain_map[i]);
	}


	gain_map.clear();

	return true;
}

bool SurroundView::stitch(const std::vector<cv::cuda::GpuMat*>& imgs)
{
    if (!isInit){
        std::cerr << "SurroundView not initialize...\n";
        return false;
    }

    cv::cuda::GpuMat gpuimg_warped_s, gpuimg_warped;

    for(size_t i = 0; i < imgs_num; ++i){

          cv::cuda::remap(*imgs[i], gpuimg_warped, texXmap[i], texYmap[i], cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);

    }


    return true;
}





void SurroundView::applyGpuCompensator(cv::cuda::GpuMat& _image, cv::cuda::GpuMat& gpu_gain_map)
{
	CV_Assert(_image.type() == CV_8UC3);

	cv::cuda::GpuMat temp;

	_image.convertTo(temp, CV_32F, streamObj);

	cv::cuda::multiply(temp, gpu_gain_map, temp, 1, CV_32F, streamObj);

	temp.convertTo(_image, CV_8U, streamObj);
}







