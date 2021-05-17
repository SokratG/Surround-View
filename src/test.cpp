#include "SurroundView.hpp"
#include "Camera.hpp"
#include <csignal>
#include "display.hpp"
#include <omp.h>


static bool finish = false;
void sig_handler(int signo)
{
	if (signo == SIGINT){
		finish = true;
		std::cout << "Signal recieved\n";
	}
}


void addCar(std::shared_ptr<View>& view_)
{
    glm::mat4 transform_car(1.f);
    transform_car = glm::translate(transform_car, glm::vec3(0.f, 0.37f, 0.f));
    transform_car = glm::rotate(transform_car, glm::radians(-90.f), glm::vec3(1.f, 0.f, 0.f));
    transform_car = glm::scale(transform_car, glm::vec3(0.0012f));

    bool ok = view_->addModel("models/Dodge Challenger SRT Hellcat 2015.obj", "shaders/modelshadervert.glsl",
                    "shaders/modelshaderfrag.glsl", transform_car);
    if (!ok)
      std::cerr << "bad add model\n";
}


int CameraCycle()
{
	cv::setNumThreads(4);
#ifndef NO_OMP
	omp_set_num_threads(omp_get_max_threads());
#endif

	SyncedCameraSource source;

	cv::Size cameraSize(CAMERA_WIDTH, CAMERA_HEIGHT);
	//cv::Size cameraSize(1280, 720);
	cv::Size undistSize(640, 480);
	//cv::Size undistSize(1280, 720);

	source.setFrameSize(cameraSize);

	int code = source.init("calibrationData/video", undistSize, false);
	if (code < 0){
		std::cerr << "source init failed " << code << "\n";
		return code;
	}	
	
	source.startStream();
	std::shared_ptr<View> view_scene = std::make_shared<View>();
	std::shared_ptr<DisplayView> dp = std::make_shared<DisplayView>();


	std::array<SyncedCameraSource::Frame, 4> frames;

	std::string win1{"Cam0"};
	std::string win2{"Cam1"};

        //cv::VideoWriter invid("stream.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, cameraSize);
	
	//cv::namedWindow(win1, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
	//cv::namedWindow(win2, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);


	SurroundView sv;
	
	auto lastTick = std::chrono::high_resolution_clock::now();
	for (; !finish; ){			
	
		if (!source.capture(frames)){
			std::cerr << "capture failed\n"; 	
			std::this_thread::sleep_for(1ms); 
			continue;
		}	
//#define YES
#define GL_YES
#ifdef YES
		cv::imshow(win1, frames[0].gpuFrame);
		cv::imshow(win2, frames[1].gpuFrame);
#else
		if (!sv.getInit()){
			std::vector<cv::cuda::GpuMat> datas {frames[0].gpuFrame, frames[1].gpuFrame, frames[2].gpuFrame, frames[3].gpuFrame};
			//auto init = sv.init(datas);
			auto init = sv.initFromFile("campar/", datas, true);
#ifdef GL_YES
			if (init){
			    const auto tex_size = sv.getResSize();
			    dp->init(1280, 720, tex_size.width, tex_size.height, view_scene);
			    addCar(view_scene);
			}
#endif
		}
		else{
		    std::vector<cv::cuda::GpuMat*> datas {&frames[0].gpuFrame, &frames[1].gpuFrame, &frames[2].gpuFrame, &frames[3].gpuFrame};
		    cv::cuda::GpuMat res;

		    sv.stitch(datas, res);
		    //cv::imshow(win1, res);
#ifdef GL_YES
		    bool okRender = dp->render(res);
		    if (!okRender)
		      break;
#endif
		}
#endif


#ifndef GL_YES
		if (cv::waitKey(1) > 0)
			break;
#endif
		const auto now = std::chrono::high_resolution_clock::now();
		const auto dt = now - lastTick;
		lastTick = now;
		const int dtMs = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
		std::cout << "dt = " << dtMs << " ms\n";
	}
	 	
	 
	source.stopStream();	
	
	return 0;
}

int main(int argc, char* argv[])
{

	signal(SIGINT, sig_handler);
	std::cout << "Started\n";
	int c = CameraCycle();
	
	
	std::cout << "Done!\n";
	return 0;
}





