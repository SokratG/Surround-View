#include <string>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>


#include <utility>
#include <algorithm>
#include <iterator>
#include <cmath>


#include <cuda_runtime.h>

#include "customBlender.h"

#include <thread>
#include <chrono>

#define MAX_MASK_WIDTH 1500
#define MAX_MASK_HEIGHT 1024
#define MIN_MASK_WIDTH 320
#define MIN_MASK_HEIGHT 240

using namespace std::literals::chrono_literals;



class SurroundView
{
private:
	bool isInit = false;
	size_t imgs_num = 0;
	double warped_image_scale = 1.0;
	double work_scale = 1;
        double registr_resol = 0.7;
        int num_bands = 3;
	bool work_set = false;
	std::vector<cv::Mat> Ks_f;
	std::vector<cv::detail::CameraParams> cameras;
	/* optional */
	std::vector<cv::cuda::GpuMat> gpu_seam_masks;
	std::vector<cv::cuda::GpuMat> gpu_gain_map;
	std::vector<cv::Point> corners;
	std::vector<cv::Size> sizes;
	cv::Ptr<cv::detail::ExposureCompensator> compens;
        std::vector<cv::cuda::GpuMat> texXmap; // texture remap x-coord
        std::vector<cv::cuda::GpuMat> texYmap; // texture remap y-coord
	//
	cv::cuda::Stream streamObj;
        cv::Size mask_maxnorm_size, mask_minnorm_size;
protected:
        bool warpImage(const std::vector<cv::Mat>& imgs);
        bool prepareGainMatrices(const std::vector<cv::UMat>& warp_imgs);
        void applyGpuCompensator(cv::cuda::GpuMat& _image, cv::cuda::GpuMat& gpu_gain_map);
public:
        void setMaxNormSizeMask(cv::Size& size){mask_maxnorm_size = size;}
        cv::Size getMaxNormSizeMask() const {return mask_maxnorm_size;}
        void setMinNormSizeMask(cv::Size& size){mask_minnorm_size = size;}
        cv::Size getMinNormSizeMask() const {return mask_minnorm_size;}
        void setNumBands(const int numbands){num_bands = numbands;}
        int getNumBands() const {return num_bands;}
        bool getInit() const {return isInit;}
public:
        SurroundView() : mask_maxnorm_size(MAX_MASK_WIDTH, MAX_MASK_HEIGHT), mask_minnorm_size(MIN_MASK_WIDTH, MIN_MASK_HEIGHT) {}
	bool init(const std::vector<cv::cuda::GpuMat>& imgs, const std::vector<cv::Mat>& intrisicMat);
        bool stitch(const std::vector<cv::cuda::GpuMat*>& imgs);
};








