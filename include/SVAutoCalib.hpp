#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/timelapsers.hpp>


using uchar = unsigned char;


class SVAutoCalib
{
private:
        size_t imgs_num = 0;
        double conf_thresh;
        double match_conf;
        int maxpoints;
        double work_scale = 1;
        double warped_image_scale = 1.;
        size_t patch_size;
        size_t pyr_levels = 0;
        size_t threshold_features = 0;
        std::vector<cv::Mat> R;
        std::vector<cv::Mat> T;
        std::vector<cv::Mat> Ks_f;
        bool isInit = false;
private:
        bool computeImageFeaturesAndMatches_(const std::vector<cv::Mat>& imgs, std::vector<cv::detail::MatchesInfo>& pairwise_matches, std::vector<cv::detail::ImageFeatures>& features);
        bool computeCameraParameters(const std::vector<cv::detail::ImageFeatures>& features, const std::vector<cv::detail::MatchesInfo>& pairwise_matches);
        void saveData(const std::string& = std::string()) const;
public:
        SVAutoCalib(const size_t num_imgs, const size_t patch_size_ = 24, const size_t pyr_levels_ = 5, const size_t threshold_features_ = 32,
                  const double conf_thresh_=0.85, const double match_conf_=0.65, const int maxpoints_=2500)
            : imgs_num(num_imgs), conf_thresh(conf_thresh_), match_conf(match_conf_), maxpoints(maxpoints_),
              patch_size(patch_size_), pyr_levels(pyr_levels_), threshold_features(threshold_features_)
        {assert(num_imgs > 0);}

        void setConfThresh(const double conf_thresh_) {conf_thresh = conf_thresh_;}
        double getConfThresh() const {return conf_thresh;}
        void setMatchConf(const double match_conf_) {match_conf = match_conf_;}
        double getMatchConf() const {return match_conf;}
        void setMaxPoints(const int maxpoints_) {maxpoints = maxpoints_;}
        double getMaxPoints() const {return maxpoints;}

        void setPatchSize(const size_t patch_size_) {patch_size = patch_size_;}
        size_t getPatchSize() const {return patch_size;}
        void setPyrLevels(const size_t pyr_levels_) {pyr_levels = pyr_levels_;}
        size_t getPyrLevels() const {return pyr_levels;}
        void setThreshFeatures(const size_t threshold_features_) {threshold_features = threshold_features_;}
        size_t getThreshFeatures() const {return threshold_features;}

public:
        bool calibrate(const std::vector<cv::Mat>& rescale_imgs, const bool savedata=false);

        std::vector<cv::Mat> getExtTranslation() const {return T;}
        std::vector<cv::Mat> getExtRotation() const{return R;}
        std::vector<cv::Mat> getIntCameraParam() const {return Ks_f;}
        double get_warpImgScale() const {return warped_image_scale;}
        cv::Mat getKf(size_t idx) const { if (isInit) return Ks_f[idx]; else return cv::Mat();}
        cv::Mat getR(size_t idx) const { if (isInit) return R[idx]; else return cv::Mat();}
};
