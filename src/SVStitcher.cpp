#include <iostream>
#include "SVAutoCalib.hpp"
#include "SVSeamDetection.hpp"
#include "SVStitcher.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


#include <omp.h>



bool SVStitcher::init(const std::vector<cv::cuda::GpuMat>& imgs){
	
	if (isInit){
	    std::cerr << "SVStitcher already initialize...\n";
	    return false;
	}
	
	if (imgs.size() <= 1){
	    std::cerr << "Error pass images - size must be greater 2...\n";
	    return false;
	}

	std::vector<cv::cuda::GpuMat> imgs_ = imgs;

	splitRearView(imgs_);

	imgs_num = imgs_.size();

	std::vector<cv::Mat> cpu_imgs(imgs_num);
	for (size_t i = 0; i < imgs_num; ++i){
	    imgs_[i].download(cpu_imgs[i]);
	    cv::resize(cpu_imgs[i], cpu_imgs[i], cv::Size(), scale_factor, scale_factor);
	}


	AutoCalib autcalib(imgs_num);
	bool res = autcalib.init(cpu_imgs, true);
	if (!res){
	    std::cerr << "Error can't autocalibrate camera parameters...\n";
	    return false;
	}
	float warped_image_scale = autcalib.get_warpImgScale();
	std::vector<cv::Mat> R = autcalib.getExtRotation();
	std::vector<cv::Mat> Ks_f = autcalib.getIntCameraParam();


	SVSeamDetector smd(imgs_num, warped_image_scale, scale_factor);
	res = smd.init(cpu_imgs, Ks_f, R);
	if (!res){
	    std::cerr << "Error can't seam masks for images...\n";
	    return false;
	}
	gpu_seam_masks = smd.getSeams();
	corners = smd.getCorners();
	sizes = smd.getSizes();
	texXmap = smd.getXmap();
	texYmap = smd.getYmap();

	if (cuBlender.get() == nullptr){
	    cuBlender = std::make_shared<SVMultiBandBlender>(numbands);
	    cuBlender->prepare(corners, sizes, gpu_seam_masks);
	}


        res = prepareCutOffFrame(cpu_imgs);
        if (!res){
            std::cerr << "Error can't prepare blending ROI rect...\n";
            return false;
        }


        gpu_warped_ = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));
        gpu_warped_s_ = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));
        gpu_warped_scale_ = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));

	isInit = true;

	return isInit;
}


bool SVStitcher::initFromFile(const std::string& dirpath, const std::vector<cv::cuda::GpuMat>& imgs, const bool use_filewarp_pts)
{
    if (isInit){
        std::cerr << "SurroundView already initialize...\n";
        return false;
    }

    if (dirpath.empty()){
        std::cerr << "Invalid directory path...\n";
        return false;
    }

    if (imgs.size() <= 1){
        std::cerr << "Error pass images - size must be greater 2...\n";
        return false;
    }

    std::vector<cv::cuda::GpuMat> imgs_ = imgs;

    splitRearView(imgs_);

    imgs_num = imgs_.size();

    std::vector<cv::Mat> cpu_imgs(imgs_num);
    for (size_t i = 0; i < imgs_num; ++i){
        imgs_[i].download(cpu_imgs[i]);
        cv::resize(cpu_imgs[i], cpu_imgs[i], cv::Size(), scale_factor, scale_factor);
    }


    float warped_image_scale = 1.0;
    std::vector<cv::Mat> R;
    std::vector<cv::Mat> Ks_f;
    bool res = getDataFromFile(dirpath, Ks_f, R, warped_image_scale, use_filewarp_pts);
    if (!res){
        std::cerr << "Error can't read camera parameters...\n";
        return false;
    }

    SVSeamDetector smd(imgs_num, warped_image_scale, scale_factor);
    res = smd.init(cpu_imgs, Ks_f, R);
    if (!res){
        std::cerr << "Error can't seam masks for images...\n";
        return false;
    }

    gpu_seam_masks = smd.getSeams();
    corners = smd.getCorners();
    sizes = smd.getSizes();
    texXmap = smd.getXmap();
    texYmap = smd.getYmap();


    if (cuBlender.get() == nullptr){
        cuBlender = std::make_shared<SVMultiBandBlender>(numbands);
        cuBlender->prepare(corners, sizes, gpu_seam_masks);
    }

    if (!use_filewarp_pts){
        res = prepareCutOffFrame(cpu_imgs);
        if (!res){
            std::cerr << "Error can't prepare blending ROI rect...\n";
            return false;
        }
    }

    gpu_warped_ = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));
    gpu_warped_s_ = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));
    gpu_warped_scale_ = std::move( std::vector<cv::cuda::GpuMat>(imgs_num));


    isInit = true;


    return isInit;
}


bool SVStitcher::getDataFromFile(const std::string& dirpath, std::vector<cv::Mat>& Ks_f, std::vector<cv::Mat>& R, float& warp_scale, const bool use_filewarp_pts)
{
    Ks_f = std::move(std::vector<cv::Mat>(imgs_num));
    R = std::move(std::vector<cv::Mat>(imgs_num));
    auto fullpath = dirpath + "Camparam";
    for(auto i = 0; i < imgs_num; ++i){
           std::string KRpath{fullpath + std::to_string(i) + ".yaml"};
           cv::FileStorage KRfout(KRpath, cv::FileStorage::READ);
           warp_scale = KRfout["FocalLength"];
           if (!KRfout.isOpened()){
               std::cerr << "Error can't open camera param file: " << KRpath << "...\n";
               return false;
           }
           cv::Mat_<float> K_;
           KRfout["Intrisic"] >> K_;
           K_.convertTo(Ks_f[i], CV_32F);
           KRfout["Rotation"] >> R[i];
    }

    if (use_filewarp_pts){
          std::string WARP_PTS_path{dirpath + "corner_warppts.yaml"};
          cv::Point tl, tr, bl, br;
          cv::FileStorage WPTSfout(WARP_PTS_path, cv::FileStorage::READ);
          if (!WPTSfout.isOpened()){
              std::cerr << "Error can't open camera param file: " << WARP_PTS_path << "...\n";
              return false;
          }
          WPTSfout["tl"] >> tl; WPTSfout["tr"] >> tr;
          WPTSfout["bl"] >> bl; WPTSfout["br"] >> br;
          WPTSfout["res_size"] >> resSize;
          const auto width_ = resSize.width;
          const auto height_ = resSize.height;


          std::vector<cv::Point_<float>> src {tl, tr, bl, br};
          std::vector<cv::Point_<float>> dst {cv::Point(0, 0), cv::Point(width_, 0),
                                              cv::Point(0, height_), cv::Point(width_, height_)};
          cv::Mat transformM = cv::getPerspectiveTransform(src, dst);

          cv::cuda::buildWarpPerspectiveMaps(transformM, false, resSize, warpXmap, warpYmap);

          row_range = cv::Range(tl.y, height_);
          col_range = cv::Range(0,  width_);
    }



    return true;
}


void SVStitcher::detectCorners(const cv::Mat& src, cv::Point& tl, cv::Point& bl, cv::Point& tr, cv::Point& br)
{
      std::vector<std::vector<cv::Point>> cnts;

      cv::findContours(src, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

      const auto height_ = src.rows;

      tl = cnts[0][0];
      tr = cv::Point(0, 0);
      bl = cnts[0][0];
      br = cnts[0][0];
      const auto x_constrain_tl = sizes[sizes.size() - 1].width >> 1;

      /* find bottom-left and bottorm-right corners (or if another warping tl and tr)*/
      auto idx_tl = 0, idx_tr = 0, tot_idx = 0;
      for(const auto& pcnt : cnts){
          for (const auto& pt : pcnt){
              if (bl.x >= pt.x){
                bl = pt;
              }

              if (br.x < pt.x ){
                br = pt;
                idx_tr = tot_idx;
              }

              tot_idx += 1;
              if (pt.x == x_constrain_tl && pt.y < (height_ / 2))
                idx_tl = tot_idx;
          }
      }


      /* find top-right*/
      for(auto i = idx_tr; i < cnts[0].size() - 1; ++i){
          const auto& pt = cnts[0][i];
          const auto& next_pt = cnts[0][i + 1];
          auto dy = next_pt.y - pt.y;
          if (br.y > pt.y && dy == 0){
            tr = pt;
            break;
          }
      }

      /* find top-left*/
      for(auto i = idx_tl; i < cnts[0].size() - 1; ++i){
          const auto& pt = cnts[0][i];
          const auto& next_pt = cnts[0][i + 1];
          auto dx = cnts[0][i + 2].x - next_pt.x;
          auto dy = next_pt.y - pt.y;
          if (dx == 0 && dy == 0){
            tl = pt;
            break;
          }
      }

}

bool SVStitcher::prepareCutOffFrame(const std::vector<cv::Mat>& cpu_imgs)
{
          cv::cuda::GpuMat gpu_result, warp_s, warp_img;
          for(size_t i = 0; i < imgs_num; ++i){
                  gpu_result.upload(cpu_imgs[i]);
                  cv::cuda::remap(gpu_result, warp_img, texXmap[i], texYmap[i], cv::INTER_LINEAR, cv::BORDER_REFLECT, cv::Scalar(), streamObj);
                  warp_img.convertTo(warp_s, CV_16S);
                  cuBlender->feed(warp_s, gpu_seam_masks[i], i);
          }


          cuBlender->blend(gpu_result, warp_img);
          cv::Mat result, thresh;
          gpu_result.download(result);
          warp_img.download(thresh);

          //result.convertTo(result, CV_8U);

          cv::threshold(thresh, thresh, 1, 255, cv::THRESH_BINARY);

          cv::Point tl, bl, tr, br;
          detectCorners(thresh, tl, bl, tr, br);

          const auto height_ = result.rows;
          const auto width_ = result.cols;
          resSize = result.size();
          /* add offset of coordinate corner points due to seam last frame */

          save_warpptr("corner_warppts.yaml", resSize, tl, tr, bl, br);

          std::vector<cv::Point_<float>> src {tl, tr, bl, br};

          std::vector<cv::Point_<float>> dst {cv::Point(0, 0), cv::Point(width_, 0),
                                              cv::Point(0, height_), cv::Point(width_, height_)};
          cv::Mat transformM = cv::getPerspectiveTransform(src, dst);
          //cv::warpPerspective(result, result, transformM, resSize, cv::INTER_CUBIC, cv::BORDER_CONSTANT);

          cv::cuda::buildWarpPerspectiveMaps(transformM, false, resSize, warpXmap, warpYmap);

          row_range = cv::Range(tl.y, height_);
          col_range = cv::Range(0,  width_);

          return true;
}

void SVStitcher::save_warpptr(const std::string& warpfile, const cv::Size& res_size,
                                const cv::Point& tl, const cv::Point& tr, const cv::Point& bl, const cv::Point& br)
{
      cv::FileStorage WPTSfout(warpfile, cv::FileStorage::WRITE);
      WPTSfout << "res_size" << res_size;
      WPTSfout << "tl" << tl;
      WPTSfout << "tr" << tr;
      WPTSfout << "bl" << bl;
      WPTSfout << "br" << br;

}


bool SVStitcher::stitch(std::vector<cv::cuda::GpuMat>& imgs, cv::cuda::GpuMat& blend_img)
{
    if (!isInit){
        std::cerr << "SurroundView was not initialized...\n";
        return false;
    }

    splitRearView(imgs);

#ifndef NO_OMP
    #pragma omp parallel for default(none) shared(imgs)
#endif
    for(size_t i = 0; i < imgs_num; ++i){

          cv::cuda::resize(imgs[i], gpu_warped_scale_[i], cv::Size(), scale_factor, scale_factor, cv::INTER_NEAREST, loopStreamObj);

          cv::cuda::remap(gpu_warped_scale_[i], gpu_warped_[i], texXmap[i], texYmap[i], cv::INTER_NEAREST, cv::BORDER_REFLECT, cv::Scalar(), loopStreamObj);

          gpu_warped_[i].convertTo(gpu_warped_s_[i], CV_16S, loopStreamObj);

          cuBlender->feed(gpu_warped_s_[i], i, loopStreamObj);
    }

    cuBlender->blend(stitch_, streamObj);

    cv::cuda::remap(stitch_, stitch_remap_, warpXmap, warpYmap, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), streamObj);

    blend_img = stitch_remap_(row_range, col_range);

    return true;
}


void SVStitcher::splitRearView(std::vector<cv::cuda::GpuMat>& imgs)
{
    auto last_idx = imgs.size() - 1;
    auto rear_width = imgs[last_idx].cols;
    auto rear_half_width = (rear_width >> 1) + 1;
    auto rear_height = imgs[last_idx].rows;
    cv::cuda::GpuMat half_rear = imgs[last_idx](cv::Range(0, rear_height), cv::Range(rear_half_width, rear_width));
    imgs[0] = imgs[last_idx](cv::Range(0, rear_height), cv::Range(0, rear_half_width));
    imgs[last_idx] = half_rear;
}


