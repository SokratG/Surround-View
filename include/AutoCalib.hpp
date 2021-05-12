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
        double conf_thresh = 0.8;
        double match_conf = 0.55;
        int maxpoints = 1000;
        double work_scale = 1;
        double warped_image_scale = 1.;
        std::vector<cv::Mat> R;
        std::vector<cv::Mat> T;
        std::vector<cv::Mat> Ks_f;
        bool isInit = false;
private:
        bool computeImageFeaturesAndMatches_(const std::vector<cv::Mat>& imgs, std::vector<cv::detail::MatchesInfo>& pairwise_matches, std::vector<cv::detail::ImageFeatures>& features);
        bool computeCameraParameters(const std::vector<cv::detail::ImageFeatures>& features, const std::vector<cv::detail::MatchesInfo>& pairwise_matches);
        void saveData(const std::string& = std::string()) const;
public:
        AutoCalib(const size_t num_imgs) : imgs_num(num_imgs) {assert(num_imgs > 0);}
        bool init(const std::vector<cv::Mat>& rescale_imgs, const bool savedata=false);

        std::vector<cv::Mat> getExtTranslation() const {return T;}
        std::vector<cv::Mat> getExtRotation() const{return R;}
        std::vector<cv::Mat> getIntCameraParam() const {return Ks_f;}
        double get_warpImgScale() const {return warped_image_scale;}
        cv::Mat getKf(size_t idx) const { if (isInit) return Ks_f[idx]; else return cv::Mat();}
        cv::Mat getR(size_t idx) const { if (isInit) return R[idx]; else return cv::Mat();}
};
