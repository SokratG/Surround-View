#include "AutoCalib.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/warpers.hpp>


#include <iostream>


bool AutoCalib::init(const std::vector<cv::Mat>& imgs)
{
	if (isInit){
		std::cerr << "Autocalibrator already initialize...\n";
		return false;
	}

	if (imgs.size() > imgs_num)
		return false;

	cv::Size img_size = imgs[0].size();


	std::vector<cv::detail::ImageFeatures> features(imgs_num);
	std::vector<cv::detail::MatchesInfo> pairwise_matches;

	if (!computeImageFeaturesAndMatches_(imgs, pairwise_matches, features)){
		std::cerr << "Error can't find pairwise features...\n";
		return false;
	}

	std::vector<int> indxs = cv::detail::leaveBiggestComponent(features, pairwise_matches, conf_thresh);


	if (pairwise_matches.size() < (imgs_num*imgs_num)){
		std::cout << pairwise_matches.size() << "\n";
		std::cerr << "Error not enough calibrates images...\n";
		return false;
	}


	if (!computeCameraParameters(features, pairwise_matches))
		return false;


	isInit = true;

	return isInit;
}


bool AutoCalib::computeImageFeaturesAndMatches_(const std::vector<cv::Mat>& imgs, std::vector<cv::detail::MatchesInfo>& pairwise_matches, std::vector<cv::detail::ImageFeatures>& features)
{
	cv::Ptr<cv::Feature2D> finder = cv::ORB::create(maxpoints, 1.2, 9, 27, 0, 3, cv::ORB::HARRIS_SCORE, 31, 20);
	cv::Ptr<cv::detail::FeaturesMatcher> matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true, match_conf);

	for (int i = 0; i < imgs_num; ++i)
		cv::detail::computeImageFeatures(finder, imgs[i], features[i]);


	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();
	return true;
}

bool AutoCalib::computeCameraParameters(const std::vector<cv::detail::ImageFeatures>& features, const std::vector<cv::detail::MatchesInfo>& pairwise_matches)
{

	cv::detail::HomographyBasedEstimator est;

	if (!est(features, pairwise_matches, cameras)){
		std::cerr << "Error refinement camera params...\n";
		return false;
	}

	for (size_t i = 0; i < cameras.size(); i++){
		cv::Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
#ifdef NO_COMPILE
		cameras[i].focal = intrisicMat[i].at<double>(0, 0);
		cameras[i].ppx = intrisicMat[i].at<double>(0, 2);
		cameras[i].ppy = intrisicMat[i].at<double>(1, 2);
#endif
	}

	cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();

	adjuster->setConfThresh(conf_thresh);

	if (!(*adjuster)(features, pairwise_matches, cameras)){
		std::cerr << "Error refinement camera params...\n";
		return false;
	}


	std::vector<cv::Mat> rmats;
	for(size_t i = 0; i < cameras.size(); ++i)
		rmats.emplace_back(cameras[i].R.clone());

	std::vector<float> focals;
	cv::detail::waveCorrect(rmats, cv::detail::WAVE_CORRECT_HORIZ);
	for(size_t i = 0; i < cameras.size(); ++i){
		cameras[i].R = rmats[i];
		focals.emplace_back(cameras[i].focal);
		cv::Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		Ks_f.emplace_back(K);
	}

	std::sort(focals.begin(), focals.end());
	size_t focals_size = focals.size();
	if (focals_size % 2)
		warped_image_scale = static_cast<float>(focals[focals_size / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals_size / 2 - 1] + focals[focals_size / 2]) * 0.5f;

	return true;
}

