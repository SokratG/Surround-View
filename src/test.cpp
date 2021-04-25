#include "Camera.hpp"
#include "SurroundView.hpp"
#include <csignal>
#include "display.hpp"



static bool finish = false;
void sig_handler(int signo)
{
	if (signo == SIGINT){
		finish = true;
		std::cout << "Signal recieved\n";
	}
}


int CameraCycle()
{
	cv::setNumThreads(4);
	SyncedCameraSource source;

	cv::Size cameraSize(CAMERA_WIDTH, CAMERA_HEIGHT);
	//cv::Size cameraSize(1280, 720);
	cv::Size undistSize(1280, 720);
	
	source.setFrameSize(cameraSize);

	int code = source.init("calibrationData/video", undistSize, false);
	if (code < 0){
		std::cerr << "source init failed " << code << "\n";
		return code;
	}	
	
	source.startStream();

	// std::vector<cv::Mat> Ks{source.Ks[0], source.Ks[1], source.Ks[2], source.Ks[3]};
	std::vector<cv::Mat> Ks{source.Ks[1], source.Ks[2]};
	
	std::shared_ptr<View> view_scene = std::make_shared<View>();
	DisplayView dp;
	//dp.init(1280, 720, view_scene);

	std::array<SyncedCameraSource::Frame, 4> frames;
	std::string win1{"Camera - 1:"};	
	std::string win2{"Camera - 2:"};


	
        //cv::VideoWriter invid("stream.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, cameraSize);
	
	cv::namedWindow(win1, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
	//cv::namedWindow(win2, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);


	SurroundView sv;
	
	auto lastTick = std::chrono::high_resolution_clock::now();
	for (; !finish; ){			
	
		if (!source.capture(frames)){
			std::cerr << "capture failed\n"; 	
			std::this_thread::sleep_for(1ms); 
			continue;
		}	
		
				
		if (!sv.getInit()){
			std::vector<cv::cuda::GpuMat> datas {frames[1].gpuFrame, frames[2].gpuFrame};
			sv.init(datas, Ks);
			//cv::imshow(win1, frames[2].gpuFrame);
			//cv::imshow(win2, frames[2].gpuFrame);
		}
		else{
		    std::vector<cv::cuda::GpuMat*> datas {&frames[1].gpuFrame, &frames[2].gpuFrame};
		    cv::Mat res;
		    sv.stitch(datas, res);
		    cv::imshow(win1, res);
		}


/*
		bool okRender = dp.render(frames);
		if (!okRender)
		  break;
*/
		

		if (cv::waitKey(1) > 0)
			break;


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





