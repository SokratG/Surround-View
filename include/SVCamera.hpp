#pragma once
#include <iostream>
#include <string>

#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudacodec.hpp>

#include <cuda_runtime.h>

#include <linux/videodev2.h>

using namespace std::literals::chrono_literals;


using uchar = unsigned char;

/*
    Video4linux 2
*/

#define CAMERA_WIDTH 1280
#define CAMERA_HEIGHT 720

#define BUFFER_MAP
#define MMAP_BUFFERS_COUNT 4
#define CAM_NUMS 4

class CameraInfo
{
#ifdef BUFFER_MAP
private:	
	typedef struct _buffer{
		void* start;
		size_t length;
	} buffer;
#endif
public:
	int fd = -1;
	std::string devicePath;
	uint32_t capabilities = 0;
	std::vector<v4l2_fmtdesc> formats;
	cv::Size frameSize;
	size_t frameSizeBytes = 0;
	
	bool streamStarted = false;
	
	bool cuda_zero_copy = true;
	uchar* cuda_out_buffer = nullptr;
#ifdef BUFFER_MAP	
	std::vector<buffer> buffers;
#endif
public:
	CameraInfo(const std::string& devicePath_ ={}, const cv::Size &frameSize_ = {}) : 
		devicePath(devicePath_), frameSize(frameSize_) {}
	~CameraInfo(){ deinit(); }
	

	bool init(const std::string &devicePath_ = {});
	bool deinit();
private:
	bool initFormats();
	void deinitCuda();
#ifdef BUFFER_MAP
	bool initMMap();
	void deinitMMap();
#endif
	bool initCaps();
	bool initStream();
	bool initCuda();
public:
	bool startStream();
	bool stopStream();
	bool capture(size_t timeout, cv::Mat& res) const;
};

typedef struct _InternalCameraParams
{
	cv::Size resolution;
	std::array<double, 9> K;
	std::array<double, 14> distortion;
	cv::Size captureResolution;
	bool read(const std::string& filepath, const int camNum, const cv::Size& resol = cv::Size(1920, 1080), const cv::Size& cameraResol = cv::Size(1920, 1080));
private:
	int _cameraNum;
} InternalCameraParams;

typedef struct _ExternalCameraParams
{
	//
} ExternalCameraParams;


class SyncedCameraSource
{

public:
	typedef struct _Frame {
		cv::cuda::GpuMat gpuFrame;
	} Frame;
	typedef struct _CameraUndistortData{
		cv::cuda::GpuMat remapX, remapY;
                cv::cuda::GpuMat undistFrame;
		cv::Rect roiFrame;
	} CameraUndistortData; 
        std::array<cv::Mat, CAM_NUMS> Ks;
private:

	cv::Size frameSize{CAMERA_WIDTH, CAMERA_HEIGHT};
        std::array<CameraInfo, CAM_NUMS> _cams = {{CameraInfo("/dev/video0", frameSize),
					    CameraInfo("/dev/video1", frameSize),
					    CameraInfo("/dev/video2", frameSize),
					    CameraInfo("/dev/video3", frameSize)}};
        std::array<InternalCameraParams, CAM_NUMS> camIparams;
        std::array<CameraUndistortData, CAM_NUMS> undistFrames;
	bool cuda_zero_copy = true;

public:
	
	SyncedCameraSource() = default;
	~SyncedCameraSource(){ close(); }	


        int init(const std::string& param_filepath, const cv::Size& calibSize, const cv::Size& undistSize, const bool useUndist=false);
	bool startStream();
	bool stopStream();
        bool capture(std::array<Frame, CAM_NUMS>& frames);
	
	void close(){
                for (auto i = 0; i < _cams.size(); ++i){
                    cudaFree(d_src[i]);
                }
		for (auto& cam : _cams)
			cam.stopStream();

                if (_cudaStream)
                        cudaStreamDestroy(_cudaStream);
                _cudaStream = NULL;
	}
public:
	const CameraInfo& getCamera(int index) const { return _cams[index]; }
	size_t getCamerasCount() const { return _cams.size(); }
	cv::Size getFramesize() const { return frameSize; }
	bool setFrameSize(const cv::Size& size); 
	bool _undistort = true;
	const CameraUndistortData& getUndistortData(const size_t idx) const{
		return undistFrames[idx];
	}
	
private:
        std::array<v4l2_buffer, CAM_NUMS> buffs{};
        cudaStream_t _cudaStream {NULL};
        cv::cuda::Stream cudaStreamObj{cv::cuda::Stream::Null()};
        uchar* d_src[4]; // cuda source memory
};




















