#include <string>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <utility>
#include <algorithm>
#include <iterator>
#include <cmath>


#include <cuda_runtime.h>

#include "customBlender.h"




#define CUT_OFF_FRAME
class SurroundView
{
private:
        static constexpr auto threshold_color = 127;
private:
	bool isInit = false;
	size_t imgs_num = 0;
	double warped_image_scale = 1.0;
        double sharpness = 2.5f;
	std::vector<cv::Mat> Ks_f;
	std::vector<cv::detail::CameraParams> cameras;
	std::vector<cv::cuda::GpuMat> gpu_seam_masks;
	std::vector<cv::Point> corners;
	std::vector<cv::Size> sizes;
        cv::Range blendingEdges;
        cv::Size resSize;
        /* optional */
        std::vector<cv::cuda::GpuMat> texXmap; // texture remap x-coord
        std::vector<cv::cuda::GpuMat> texYmap; // texture remap y-coord
        // --------------
	cv::cuda::Stream streamObj;
        std::shared_ptr<CUDAFeatherBlender> cuBlender;
#ifdef COLOR_CORRECTION
        std::vector<cv::cuda::GpuMat> inrgb = std::move(std::vector<cv::cuda::GpuMat>(3));
#endif
#ifdef CUT_OFF_FRAME
private:
        bool prepareCutOffFrame(const std::vector<cv::Mat>& cpu_imgs);
#endif
public:
        bool getInit() const {return isInit;}
        cv::Size getResSize() const {return resSize;}
public:
        SurroundView() :cuBlender(nullptr) {}
        bool init(const std::vector<cv::cuda::GpuMat>& imgs);
        bool stitch(const std::vector<cv::cuda::GpuMat*>& imgs, cv::cuda::GpuMat& blend_img);
};








