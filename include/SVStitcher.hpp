#pragma once
#include <string>
#include <utility>
#include <cmath>

#include <opencv2/cudaimgproc.hpp>

#include <SVGainCompensator.hpp>
#include <SVBlender.hpp>


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
        cv::cuda::GpuMat half_rear;
        // --------------
        cv::cuda::Stream streamObj, loopStreamObj, photoStreamObj;
        std::shared_ptr<SVMultiBandBlender> cuBlender;
        std::shared_ptr<SVChannelCompensator> svGainComp;
        // --------------
        std::vector<cv::cuda::GpuMat> warp_gain_gpu, gpu_scale;
        std::vector<cv::cuda::GpuMat> vecYCrCb;
        cv::cuda::GpuMat gpu_lum_gray, lum_mean_std, log_lum_map;
        float white_luminance, tonemap_luminance;
        // --------------
        bool isInit = false;
private:
        cv::cuda::GpuMat stitch_, stitch_ROI_;
        std::vector<cv::cuda::GpuMat> gpu_warped_, gpu_warped_s_, gpu_warped_scale_;
private:
        void save_warpptr(const std::string& warpfile, const cv::Size& res_size,
                          const cv::Point& tl, const cv::Point& tr, const cv::Point& bl, const cv::Point& br);
        bool prepareCutOffFrame(const std::vector<cv::Mat>& cpu_imgs);
        bool getDataFromFile(const std::string& dirpath, std::vector<cv::Mat>& Ks_f, std::vector<cv::Mat>& R, float& warp_scale, const bool use_filewarp_pts);
        void splitRearView(std::vector<cv::cuda::GpuMat>& imgs);
        void detectCorners(const cv::Mat& src, cv::Point& tl, cv::Point& bl, cv::Point& tr, cv::Point& br);
        void computeGains(const std::vector<cv::cuda::GpuMat>& gpu_imgs, const std::vector<cv::cuda::GpuMat>& gpu_warped_mask);
        void computeMaxLuminance(const cv::cuda::GpuMat& img);
        /* //avoid alloc-dealloc, try implement cuda split rear view
         * void cuSplitRearView(std::vector<cv::cuda::GpuMat>& imgs);
        */
public:
        bool getInit() const {return isInit;}
        cv::Size getResSize() const {return resSize;}
        void setNumbands(const size_t numbands_){ numbands = numbands_;}
        size_t getNumbands() const { return numbands;}
        float getLuminance() const {return tonemap_luminance;}
        float getWhiteLuminance() const {return white_luminance;}
public:
        SVStitcher(const size_t numbands_ = 4, const float scale_factor_ = 1.0) :
            cuBlender(nullptr), numbands(numbands_), scale_factor(scale_factor_),
            white_luminance(1.0), tonemap_luminance(1.0)
        {}
        bool init(const std::vector<cv::cuda::GpuMat>& imgs);
        bool initFromFile(const std::string& dirpath, const std::vector<cv::cuda::GpuMat>& imgs, const bool use_filewarp_pts=false);
        bool stitch(std::vector<cv::cuda::GpuMat>& imgs, cv::cuda::GpuMat& blend_img);
        void recomputeGain(const std::vector<cv::cuda::GpuMat>& gpu_imgs);
        void recomputeToneLuminance(const cv::cuda::GpuMat& img);
};








