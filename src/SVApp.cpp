#include <SVApp.hpp>

#include <csignal>

#include <omp.h>

#define GL_USE

#ifndef GL_USE
#include <opencv2/highgui.hpp>
#endif


#define LOG_USE

static bool finish = false;
void sig_handler(int signo)
{
        if (signo == SIGINT){
                finish = true;
                std::cout << "Signal recieved\n";
        }
}



static void addCar(std::shared_ptr<SVRender>& view_, const SVAppConfig& svcfg)
{
    glm::mat4 transform_car(1.f);
    transform_car = glm::translate(transform_car, glm::vec3(0.f, 0.96f, 0.f));
    transform_car = glm::rotate(transform_car, glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
    transform_car = glm::scale(transform_car, glm::vec3(0.0016f));

    bool is_Add = view_->addModel(svcfg.car_model, svcfg.car_vert_shader,
                    svcfg.car_frag_shader, transform_car);
    if (!is_Add)
      std::cerr << "Error can't add model\n";
}

static void addBowlConfig(ConfigBowl& cbowl)
{
    /* Bowl parameter */
    glm::mat4 transform_bowl(1.f);
    cbowl.transformation = transform_bowl;
    cbowl.disk_radius = 0.4f;
    cbowl.parab_radius = 0.55f;
    cbowl.hole_radius = 0.09f;
    cbowl.a = 0.4f; cbowl.b = 0.4f; cbowl.c = 0.2f;
    cbowl.vertices_num  = 750.f;
}


SVApp::SVApp(const SVAppConfig& svcfg) :
    cameraSize(svcfg.cam_width, svcfg.cam_height), undistSize(svcfg.cam_width, svcfg.cam_height),
    calibSize(svcfg.calib_width, svcfg.calib_height), limit_iteration_show(-1), threadpool(svcfg.num_pool_threads)
{
    svappcfg = svcfg;
    cameradata = std::move(std::vector<cv::cuda::GpuMat>(CAM_NUMS + 1));
}

SVApp::~SVApp()
{
    release();
}


void SVApp::release()
{
    source->stopStream();
}


bool SVApp::init(const int limit_iteration_init_)
{
        signal(SIGINT, sig_handler);

#ifndef NO_OMP
        omp_set_num_threads(omp_get_max_threads());
#endif
        limit_iteration_init = limit_iteration_init_;

        source = std::make_shared<SyncedCameraSource>();

        source->setFrameSize(cameraSize);

        int code = source->init(svappcfg.undist_folder, calibSize, undistSize, false);
        if (code < 0){
                std::cerr << "source init failed " << code << "\n";
                return false;
        }

        usePedDetect = svappcfg.usePedestrianDetection;

#ifndef GL_USE
        cv::namedWindow(svappcfg.win1, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
        cv::namedWindow(svappcfg.win2, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
#endif
        source->startStream();

        view_scene = std::make_shared<SVRender>(cameraSize.width, cameraSize.height);
        dp = std::make_shared<SVDisplayView>();
        svtitch = std::make_shared<SVStitcher>(svappcfg.numbands, svappcfg.scale_factor);
        if (usePedDetect)
            sv_ped_det = std::make_shared<SVPedDetect>();

        //cv::VideoWriter invid("stream.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, cameraSize);

        auto init = false;
        while (!svtitch->getInit() && limit_iteration_init != 0){

                if (!source->capture(frames)){
                        std::cerr << "capture failed\n";
                        std::this_thread::sleep_for(1ms);
                        continue;
                }
#ifndef GL_USE
                //cv::imshow(svappcfg.win1, frames[0].gpuFrame);
                //cv::imshow(svappcfg.win2, frames[1].gpuFrame);
#endif
                std::vector<cv::cuda::GpuMat> datas {cv::cuda::GpuMat(), frames[0].gpuFrame, frames[1].gpuFrame, frames[2].gpuFrame, frames[3].gpuFrame};
                //auto init = sv.init(datas);
                init = svtitch->initFromFile(svappcfg.calib_folder, datas, false);
#ifdef GL_USE
                if (init){
                    addBowlConfig(svappcfg.cbowl);
                    dp->init(cameraSize.width, cameraSize.height, view_scene);

                    view_scene->init(svappcfg.cbowl, svappcfg.surroundshadervert, svappcfg.surroundshaderfrag,
                                     svappcfg.screenshadervert, svappcfg.screenshaderfrag);

                    addCar(view_scene, svappcfg);
                }
#else
                if (cv::waitKey(1) > 0)
                    break;
#endif
                if (limit_iteration_init > 0)
                    limit_iteration_init -= 1;
        }

        return init;
}


void SVApp::run()
{
    auto lastTick = std::chrono::high_resolution_clock::now();
    time_recompute_gain = 0;
    for (; !finish; ){

            if (!source->capture(frames)){
                    std::cerr << "capture failed\n";
                    std::this_thread::sleep_for(1ms);
                    continue;
            }

            for (auto i = 1; i <= frames.size(); ++i)
              cameradata[i] = frames[i -1].gpuFrame;

            svtitch->stitch(cameradata, stitch_frame);

            if (usePedDetect)
                sv_ped_det->detect(stitch_frame, pedestrian_rect);


#ifdef GL_USE
            view_scene->setLuminance(svtitch->getLuminance());

            bool okRender = dp->render(stitch_frame);
            if (!okRender)
              break;
            std::this_thread::sleep_for(3ms);
#else
           cv::imshow(svappcfg.win1, stitch_frame);
           if (cv::waitKey(1) > 0)
               break;
#endif

            const auto now = std::chrono::high_resolution_clock::now();
            const auto dt = now - lastTick;
            lastTick = now;
            const int dtMs = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
            eventTask(dtMs, cameradata, stitch_frame);
#ifdef LOG_USE
            std::cout << "dt = " << dtMs << " ms\n";
#endif
    }

}



void SVApp::eventTask(int dtms, const std::vector<cv::cuda::GpuMat>& datas, const cv::cuda::GpuMat& stitched_img)
{
    time_recompute_gain += dtms;
    if (std::chrono::milliseconds(time_recompute_gain) >= svappcfg.time_recompute_photometric_gain){
           time_recompute_gain = 0;
           threadpool.enqueue([=](){
                svtitch->recomputeGain(datas);
                std::this_thread::sleep_for(1ms);
           });
    }
    time_recompute_max_luminance += dtms;
    if (std::chrono::milliseconds(time_recompute_max_luminance) >= svappcfg.time_recompute_photometric_luminance){
           time_recompute_max_luminance = 0;
           threadpool.enqueue([=](){
                svtitch->recomputeLuminance(stitched_img);
                std::this_thread::sleep_for(1ms);
           });
    }

}

