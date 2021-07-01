#pragma once
#include <SVCamera.hpp>
#include <SVStitcher.hpp>
#include <SVDisplay.hpp>
#include <SVPedestrian.hpp>

#include <ThreadPool.hpp>


struct SVAppConfig
{
    int cam_width = CAMERA_WIDTH;
    int cam_height = CAMERA_HEIGHT;
    int calib_width = CAMERA_WIDTH;
    int calib_height = CAMERA_HEIGHT;
    std::string undist_folder {"calibrationData/1280/video"};
    std::string calib_folder = "campar/";
    std::string car_model = "models/Dodge Challenger SRT Hellcat 2015.obj";
    std::string car_vert_shader = "shaders/modelshadervert.glsl";
    std::string car_frag_shader = "shaders/modelshaderfrag.glsl";
    std::string win1{"Cam0"}; // window name
    std::string win2{"Cam1"};
    int numbands = 4;
    float scale_factor = 0.65;
    int limit_iteration_init = 5000;
    int num_pool_threads = 2;
    std::chrono::seconds time_recompute_photometric_gain{10};
    std::chrono::seconds time_recompute_photometric_luminance{7};
    ConfigBowl cbowl;
    std::string surroundshadervert = "shaders/surroundvert.glsl";
    std::string surroundshaderfrag = "shaders/surroundfrag.glsl";
    std::string screenshadervert = "shaders/frame_screenvert.glsl";
    std::string screenshaderfrag = "shaders/frame_screenfrag.glsl";
    bool usePedestrianDetection = false;
};


class SVApp
{
private:
    ThreadPool threadpool;
    SVAppConfig svappcfg;
    int limit_iteration_init;
    int limit_iteration_show;
    std::shared_ptr<SyncedCameraSource> source;
    cv::Size cameraSize;
    cv::Size undistSize;
    cv::Size calibSize;
private:
    std::shared_ptr<SVRender> view_scene;
    std::shared_ptr<SVDisplayView> dp;
    std::shared_ptr<SVStitcher> svtitch;
    std::shared_ptr<SVPedDetect> sv_ped_det;
    std::array<SyncedCameraSource::Frame, CAM_NUMS> frames;
    std::vector<cv::cuda::GpuMat> cameradata;
    std::vector<cv::Rect> pedestrian_rect;
    cv::cuda::GpuMat stitch_frame;
    int time_recompute_gain, time_recompute_max_luminance;
    bool usePedDetect;
protected:
    void release();
    void eventTask(int dtms, const std::vector<cv::cuda::GpuMat>& datas, const cv::cuda::GpuMat& stitched_img);
public:
    SVApp(const SVAppConfig& svcfg);
    ~SVApp();

    SVApp& operator=(const SVApp&) = delete;
    SVApp(const SVApp&) = delete;

    bool init(const int limit_iteration_init_ = 5000);
    void run();

    void setLimitIterShow(const int lim_show){limit_iteration_show = lim_show;}
    int getLimitIterShow() const {return limit_iteration_show;}
};
