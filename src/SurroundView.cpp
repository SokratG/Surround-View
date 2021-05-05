#include <iostream>
#include "AutoCalib.hpp"
#include "SeamDetection.hpp"
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


	SeamDetector smd(imgs_num, warped_image_scale);
	res = smd.init(cpu_imgs, Ks_f, cameras);
	if (!res){
	    std::cerr << "Error can't seam masks for images...\n";
	    return false;
	}
	gpu_seam_masks = smd.getSeams();
	corners = smd.getCorners();
	sizes = smd.getSizes();
	texXmap = smd.getXmap();
	texYmap = smd.getYmap();


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
          cv::threshold(thresh, thresh, 32, 255, cv::THRESH_BINARY);
          cv::Canny(thresh, thresh, 1, 255);

          cv::morphologyEx(thresh, thresh, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
          cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

          std::vector<std::vector<cv::Point>> cnts;
          cv::findContours(thresh, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
          cv::Point tl(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
          cv::Point tr(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
          cv::Point bl(std::numeric_limits<int>::max(), std::numeric_limits<int>::min());
          cv::Point br(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());

          auto width_ = result.cols;
          auto height_ = result.rows;
          const auto x_constrain = 0.02 * width_;
          const auto y_constrain = height_ - (height_ / 4);
          for(const auto& pcnt : cnts){
              for(const auto& pt : pcnt){
                if (pt.x <= x_constrain && tl.x >= pt.x && tl.y > pt.y)
                  tl = pt;
                if (pt.x >= (width_ - x_constrain) && tr.x <= pt.x && tr.y > pt.y)
                  tr = pt;
                if (pt.x <= x_constrain && bl.x >= pt.x && bl.y < pt.y)
                  bl = pt;
                if (pt.y > y_constrain && pt.x >= (width_ - x_constrain) && br.x <= pt.x && br.y < pt.y)
                  br = pt;

              }

          }

          //constexpr auto tb_remove = 0.01;
          resSize = result.size();
          blendingEdges = cv::Range(tl.y, br.y);

          std::vector<cv::Point_<float>> src {tl, tr, bl, br};
          std::vector<cv::Point_<float>> dst {cv::Point(0, 0), cv::Point(width_, 0), cv::Point(0, height_), cv::Point(width_, height_)};
          transformM = cv::getPerspectiveTransform(src, dst);

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
    /*
    stitch.convertTo(blend_img, CV_8U, streamObj);
    */
    cv::cuda::warpPerspective(stitch, temp, transformM, resSize, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);
    //temp = stitch(cv::Range(blendingEdges.start, blendingEdges.end), cv::Range(0, stitch.cols));
    temp.convertTo(blend_img, CV_8U, streamObj);
#endif
    return true;
}





