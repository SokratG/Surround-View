#include <SVGainCompensator.hpp>


#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


// ------------------------------- SVExposureCompensator --------------------------------
SVExposureCompensator::SVExposureCompensator(const size_t imgs_num_) : imgs_num(imgs_num_)
{
    warp = std::move(std::vector<cv::UMat>(imgs_num));
    mask = std::move(std::vector<cv::UMat>(imgs_num));
}

// ------------------------------- SVGainCompensator --------------------------------
SVGainCompensator::SVGainCompensator(const size_t imgs_num_, const int nr_feeds) : SVExposureCompensator(imgs_num_)
{
    compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN);
    cv::detail::GainCompensator* gain_comp = dynamic_cast<cv::detail::GainCompensator*>(compens.get());
    gain_comp->setNrFeeds(nr_feeds);
}


void SVGainCompensator::computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                                     const std::vector<cv::cuda::GpuMat>& warp_masks)
{

    for (auto i = 0; i < imgs_num; ++i){
        warp_imgs[i].download(warp[i]);
        warp_masks[i].download(mask[i]);
    }

    compens->feed(corners, warp, mask);

    std::vector<cv::Mat> gains_;
    compens->getMatGains(gains_);

    gains = cv::Mat_<double>(gains_.size(), 1);

    for (auto i = 0; i < gains_.size(); i++){
       gains(i, 0) = gains_[i].at<double>(0, 0);
    }
}


bool SVGainCompensator::apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj)
{
   if (idx > imgs_num || imgs_num <= 0)
     return false;

   cv::Scalar gain_scalar(gains(idx), gains(idx), gains(idx));

   cv::cuda::multiply(warp_img, gain_scalar, warp_img, 1, -1, streamObj);

   return true;
}




// ------------------------------- SVGainBlocksCompensator --------------------------------
SVGainBlocksCompensator::SVGainBlocksCompensator(const size_t imgs_num_, const int bl_width,
                                                 const int bl_height, const int nr_feeds) : SVExposureCompensator(imgs_num_)
{
    gain_map = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
    gain = std::move(std::vector<cv::cuda::GpuMat>(imgs_num));
    gain_channels = std::move(std::vector<cv::cuda::GpuMat>(3));
    compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS);
    cv::detail::BlocksGainCompensator* gainbl_comp = dynamic_cast<cv::detail::BlocksGainCompensator*>(compens.get());
    gainbl_comp ->setNrFeeds(nr_feeds);
    gainbl_comp ->setBlockSize(bl_width, bl_height);
}

void SVGainBlocksCompensator::computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                  const std::vector<cv::cuda::GpuMat>& warp_masks)
{

    for (auto i = 0; i < imgs_num; ++i){
        warp_imgs[i].download(warp[i]);
        warp_masks[i].download(mask[i]);
    }

    compens->feed(corners, warp, mask);

    std::vector<cv::Mat> gains_;
    compens->getMatGains(gains_);

    for (auto i = 0; i < imgs_num; ++i){
        gain_map[i].upload(gains_[i]);
        if (gain_map[i].channels() != 3){
            gain_map[i].convertTo(gain_channels[0], CV_8UC1);
            gain_map[i].convertTo(gain_channels[1], CV_8UC1);
            gain_map[i].convertTo(gain_channels[2], CV_8UC1);
            cv::cuda::merge(gain_channels, gain_map[i]);
        }
    }
}

bool SVGainBlocksCompensator::apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj)
{
    if (idx > imgs_num || imgs_num <= 0)
      return false;

    cv::cuda::resize(gain_map.at(idx), gain[idx], warp_img.size(), 0, 0, cv::INTER_LINEAR, streamObj);

    cv::cuda::multiply(warp_img, gain[idx], warp_img, 1, warp_img.type(), streamObj);

    return true;
}




// ------------------------------- SVChannelCompensator --------------------------------
SVChannelCompensator::SVChannelCompensator(const size_t imgs_num_, const int nr_feeds) : SVExposureCompensator(imgs_num_)
{
    compens = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::CHANNELS);
    cv::detail::ChannelsCompensator* ch_comp = dynamic_cast<cv::detail::ChannelsCompensator*>(compens.get());
    ch_comp->setNrFeeds(nr_feeds);
}


void SVChannelCompensator::computeGains(const std::vector<cv::Point>& corners, const std::vector<cv::cuda::GpuMat>& warp_imgs,
                  const std::vector<cv::cuda::GpuMat>& warp_masks)
{

    for (auto i = 0; i < imgs_num; ++i){
        warp_imgs[i].download(warp[i]);
        warp_masks[i].download(mask[i]);
    }

    compens->feed(corners, warp, mask);

    std::vector<cv::Mat> gains_;
    compens->getMatGains(gains_);

    gains = cv::Mat_<double>(gains_.size(), 1);

    for (auto i = 0; i < gains_.size(); i++){
       gains(i, 0) = gains_[i].at<double>(0, 0);
    }

}


bool SVChannelCompensator::apply_compensator(const int idx, cv::cuda::GpuMat& warp_img, cv::cuda::Stream& streamObj)
{
    if (idx > imgs_num || imgs_num <= 0)
      return false;

    cv::Scalar gain_scalar(gains(idx), gains(idx), gains(idx));

    cv::cuda::multiply(warp_img, gain_scalar, warp_img, 1, -1, streamObj);

    return true;
}

