#pragma once
#include <string>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <utility>
#include <algorithm>
#include <iterator>
#include <cmath>


#include <cuda_runtime.h>

#include "SVBlender.hpp"





class SVStitcher
{
private:	
        size_t imgs_num;
        float scale_factor;
        size_t numbands;
	std::vector<cv::cuda::GpuMat> gpu_seam_masks;
	std::vector<cv::Point> corners;
	std::vector<cv::Size> sizes;
        cv::Size resSize;
        /* optional */
        std::vector<cv::cuda::GpuMat> texXmap; // texture remap x-coord
        std::vector<cv::cuda::GpuMat> texYmap; // texture remap y-coord
        cv::cuda::GpuMat warpXmap, warpYmap;
        cv::Range row_range;
        cv::Range col_range;
        // --------------
        cv::cuda::Stream streamObj;
        cv::cuda::Stream loopStreamObj;
        std::shared_ptr<SVMultiBandBlender> cuBlender;
        // --------------
        bool isInit = false;
private:
        cv::cuda::GpuMat stitch_, stitch_remap_;
        std::vector<cv::cuda::GpuMat> gpu_warped_, gpu_warped_s_, gpu_warped_scale_;
private:
        void save_warpptr(const std::string& warpfile, const cv::Size& res_size,
                          const cv::Point& tl, const cv::Point& tr, const cv::Point& bl, const cv::Point& br);
        bool prepareCutOffFrame(const std::vector<cv::Mat>& cpu_imgs);
        bool getDataFromFile(const std::string& dirpath, std::vector<cv::Mat>& Ks_f, std::vector<cv::Mat>& R, float& warp_scale, const bool use_filewarp_pts);
        void splitRearView(std::vector<cv::cuda::GpuMat>& imgs);
        void detectCorners(const cv::Mat& src, cv::Point& tl, cv::Point& bl, cv::Point& tr, cv::Point& br);
        /* //avoid alloc-dealloc, try implement cuda split rear view
         * void cuSplitRearView(std::vector<cv::cuda::GpuMat>& imgs);
        */
public:
        bool getInit() const {return isInit;}
        cv::Size getResSize() const {return resSize;}
        void setNumbands(const size_t numbands_){ numbands = numbands_;}
        size_t getNumbands() const { return numbands;}
public:
        SVStitcher(const size_t numbands_ = 4, const float scale_factor_ = 1.0) :
            cuBlender(nullptr), numbands(numbands_), scale_factor(scale_factor_) {}
        bool init(const std::vector<cv::cuda::GpuMat>& imgs);
        bool initFromFile(const std::string& dirpath, const std::vector<cv::cuda::GpuMat>& imgs, const bool use_filewarp_pts=false);
        bool stitch(std::vector<cv::cuda::GpuMat>& imgs, cv::cuda::GpuMat& blend_img);
};








