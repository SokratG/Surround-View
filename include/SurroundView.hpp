#pragma once
#include <string>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <utility>
#include <algorithm>
#include <iterator>
#include <cmath>


#include <cuda_runtime.h>

#include "customBlender.h"





class SurroundView
{
private:
	bool isInit = false;
	size_t imgs_num = 0;
	double warped_image_scale = 1.0;
        int numbands = 4;
	std::vector<cv::Mat> Ks_f;
        std::vector<cv::Mat> R;
	std::vector<cv::cuda::GpuMat> gpu_seam_masks;
	std::vector<cv::Point> corners;
	std::vector<cv::Size> sizes;
        cv::Size resSize;
        /* optional */
        std::vector<cv::cuda::GpuMat> texXmap; // texture remap x-coord
        std::vector<cv::cuda::GpuMat> texYmap; // texture remap y-coord
        cv::cuda::GpuMat warpXmap, warpYmap;
        cv::Mat transformM;
        cv::Range row_range;
        cv::Range col_range;
        // --------------
        cv::cuda::Stream streamObj;
        std::shared_ptr<CUDAMultiBandBlender> cuBlender;

private:
        void save_warpptr(const std::string& warpfile, const cv::Size& res_size,
                          const cv::Point& tl, const cv::Point& tr, const cv::Point& bl, const cv::Point& br);
        bool prepareCutOffFrame(const std::vector<cv::Mat>& cpu_imgs);
        bool getDataFromFile(const std::string& dirpath, const bool use_filewarp_pts);

public:
        bool getInit() const {return isInit;}
        cv::Size getResSize() const {return resSize;}
public:
        SurroundView() : cuBlender(nullptr) {}
        bool init(const std::vector<cv::cuda::GpuMat>& imgs);
        bool initFromFile(const std::string& dirpath, const std::vector<cv::cuda::GpuMat>& imgs, const bool use_filewarp_pts=false);
        bool stitch(const std::vector<cv::cuda::GpuMat>& imgs, cv::cuda::GpuMat& blend_img);
};








