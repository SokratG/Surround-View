#include <opencv2/features2d.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>

#include <opencv2/stitching/detail/timelapsers.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>

using uchar = unsigned char;


class AutoCalib
{
private:
        size_t imgs_num = 0;
        double conf_thresh = 1.0;
        double match_conf = 0.6;
        int maxpoints = 1200;
        double work_scale = 1;
        double warped_image_scale = 1.;
        std::vector<cv::detail::CameraParams> cameras;
        std::vector<cv::Mat> Ks_f;
        bool isInit = false;
private:
        bool computeImageFeaturesAndMatches_(const std::vector<cv::Mat>& imgs, std::vector<cv::detail::MatchesInfo>& pairwise_matches, std::vector<cv::detail::ImageFeatures>& features);
        bool computeCameraParameters(const std::vector<cv::detail::ImageFeatures>& features, const std::vector<cv::detail::MatchesInfo>& pairwise_matches, const std::vector<cv::Mat>& intrisicMat);
public:
        AutoCalib(const size_t num_imgs) : imgs_num(num_imgs) {assert(num_imgs > 0);}
        bool init(const std::vector<cv::Mat>& rescale_imgs, const std::vector<cv::Mat>& intrisicMat);

        std::vector<cv::detail::CameraParams> getExtCameraParam() const {return cameras;}
        std::vector<cv::Mat> getIntCameraParam() const {return Ks_f;}
        double get_warpImgScale() const {return warped_image_scale;}
        cv::Mat getKf(size_t idx) const { if (isInit) return Ks_f[idx]; else return cv::Mat();}
        cv::Mat getR(size_t idx) const { if (isInit) return cameras[idx].R; else return cv::Mat();}
};
