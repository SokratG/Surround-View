#include <ctime>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>


#include <array>
#include <fstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>


#include <linux/videodev2.h>
#include <sys/mman.h>

#include <omp.h>

#include "cusrc/yuv2rgb.cuh"

#include "SVCamera.hpp"


#define LOG_DEBUG(msg, ...)   printf("DEBUG:   " msg "\n", ##__VA_ARGS__)
#define LOG_WARNING(msg, ...) printf("WARNING: " msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...)   printf("ERROR:   " msg "\n", ##__VA_ARGS__)


namespace ccu = cv::cuda;

static int xioctl(int fd, int request, void* arg)
{
	int status = -1;
	do{
		status = ioctl(fd, request, arg);
	}
	while((status == -1) && EINTR == errno);
	return status;
}


inline const char* v4l2_format_str(uint32_t fmt)
{
	switch(fmt)
	{
		case V4L2_PIX_FMT_SBGGR8: return "SBGGR8 (V4L2_PIX_FMT_SBGGR8)";
		case V4L2_PIX_FMT_SGBRG8: return "SGBRG8 (V4L2_PIX_FMT_SGBRG8)";
		case V4L2_PIX_FMT_SGRBG8: return "SGRBG8 (V4L2_PIX_FMT_SGRBG8)";
		case V4L2_PIX_FMT_SRGGB8: return "SRGGB8 (V4L2_PIX_FMT_SRGGB8)";
		case V4L2_PIX_FMT_SBGGR16: return "SBGGR16 (V4L2_PIX_FMT_SBGGR16)";
		case V4L2_PIX_FMT_SRGGB10: return "SRGGB10 (V4L2_PIX_FMT_SRGGB10)";
		case V4L2_PIX_FMT_UYVY: return "UYVY (V4L2_PIX_FMT_UYVY)";
	}
	return "UNKNOW";
}

inline void v4l2_print_formatdesc(const v4l2_fmtdesc& desc)
{
	LOG_DEBUG("CameraV4L2 -- format #u%", desc.index);
	LOG_DEBUG("CameraV4L2 -- desc   %s", desc.description);
	LOG_DEBUG("CameraV4L2 -- flags  %s", (desc.flags == 0 ? "V4L2_FMT_FLAG_UNCOMPRESSED" : "V4L2_FMT_FLAG_COMPRESSED"));
	LOG_DEBUG("CameraV4L2 -- fourcc 0x%X %s", desc.pixelformat, v4l2_format_str(desc.pixelformat));
}

inline void v4l2_print_format(const v4l2_format& fmt, const char* text)
{

	LOG_DEBUG("CameraV4L2 -- %s", text);
	LOG_DEBUG("CameraV4L2 -- width %u", fmt.fmt.pix.width);
	LOG_DEBUG("CameraV4L2 -- height %u", fmt.fmt.pix.height);
	LOG_DEBUG("CameraV4L2 -- pitch %u", fmt.fmt.pix.bytesperline);
	LOG_DEBUG("CameraV4L2 -- size %u", fmt.fmt.pix.sizeimage);
	LOG_DEBUG("CameraV4L2 -- format 0x%X %s", fmt.fmt.pix.pixelformat, v4l2_format_str(fmt.fmt.pix.pixelformat));
	LOG_DEBUG("CameraV4L2 -- color 0x%X", fmt.fmt.pix.colorspace);
	LOG_DEBUG("CameraV4L2 -- field 0x%X", fmt.fmt.pix.field);
}

// ------------------------SECTION--------------------------
// -----------------CameraInfo Implementation---------------

bool CameraInfo::init(const std::string& devicePath_)
{
	if (!devicePath_.empty())
		devicePath = devicePath_;
	
	if (devicePath.empty())
		return false;

	if ((fd = open(devicePath.c_str(), O_RDWR | O_NONBLOCK, 0)) < 0){
		LOG_ERROR("Camera device [%s] open failed with fd val %d", devicePath.c_str(), fd);
		assert(0);
		return false;
	}
	
	return initCaps() && initFormats() && initStream() && initCuda();
}

bool CameraInfo::deinit()
{
	stopStream();

#ifdef BUFFER_MAP

	deinitMMap();
#endif
	::close(fd);
	fd = -1;	

	deinitCuda();
	
	return true;
}


bool CameraInfo::initCaps()
{
	assert(fd > 0);
	v4l2_capability caps;
	if (xioctl(fd, VIDIOC_QUERYCAP, &caps) < 0){
		LOG_ERROR("CameraV4L2 -- failed to query caps (xioctl VIDIOC_QUERYCAP) for %s", devicePath.c_str());
		assert(0);
		return false;
	}
	capabilities = caps.capabilities;

#define PRINT_CAP(x) printf("v4l2 -- %-18s %s\n", #x, (caps.capabilities & x) ? "yes" : "no")

	PRINT_CAP(V4L2_CAP_VIDEO_CAPTURE);
	PRINT_CAP(V4L2_CAP_READWRITE);
	PRINT_CAP(V4L2_CAP_ASYNCIO);
	PRINT_CAP(V4L2_CAP_STREAMING);

	if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)){
		LOG_ERROR("CameraV4L2 -- %s is not a video capture device", devicePath.c_str());
		assert(0);
		return false;
	}
#undef PRINT_CAP
	return true;
}



bool CameraInfo::initFormats()
{
	assert(fd > 0);
	
	v4l2_fmtdesc desc;
	std::memset(&desc, 0, sizeof(v4l2_fmtdesc));
	desc.index = 0;
	desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	
	while(ioctl(fd, VIDIOC_ENUM_FMT, &desc) == 0){
		formats.push_back(desc);
		v4l2_print_formatdesc(desc);
		++desc.index;
	}
	return true;
}


bool CameraInfo::initStream()
{
	
	v4l2_format fmt;
	std::memset(&fmt, 0, sizeof(v4l2_format));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	

	if (xioctl(fd, VIDIOC_G_FMT, &fmt) < 0){
		LOG_ERROR("CameraV4L2 -- failed to get video format device (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}	

	v4l2_print_format(fmt, "preexisting format");

	v4l2_format new_fmt;
	std::memset(&new_fmt, 0, sizeof(v4l2_format));
	
	new_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	new_fmt.fmt.pix.width = fmt.fmt.pix.width;
	new_fmt.fmt.pix.height = fmt.fmt.pix.height;
	new_fmt.fmt.pix.pixelformat = fmt.fmt.pix.pixelformat;
	new_fmt.fmt.pix.field = fmt.fmt.pix.field;
	new_fmt.fmt.pix.colorspace = fmt.fmt.pix.colorspace;

	if (frameSize.empty()){
		assert(0);
		return false;
	}
	
	new_fmt.fmt.pix.width = frameSize.width;
	new_fmt.fmt.pix.height = frameSize.height;

	const int requestedFormat = 0;
	if (requestedFormat < formats.size())
		new_fmt.fmt.pix.pixelformat = formats[requestedFormat].pixelformat;

	v4l2_print_format(new_fmt, "setting new format...");
	
	if (xioctl(fd, VIDIOC_S_FMT, &new_fmt) < 0){
		LOG_ERROR("CameraV4L2 -- failed to set video format of device (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}
	
	memset(&fmt, 0, sizeof(v4l2_format));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	
	if (xioctl(fd, VIDIOC_G_FMT, &fmt) < 0){
		LOG_ERROR("CameraV4L2 -- failed to get video format of device (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}
		
	v4l2_print_format(fmt, "confirmed new format");
	
	frameSize.width = fmt.fmt.pix.width;
	frameSize.height = fmt.fmt.pix.height;
	const auto pitch = fmt.fmt.pix.bytesperline;
	const auto depth = (pitch * 8) / frameSize.width;
	frameSizeBytes = pitch * frameSize.height;

#ifdef BUFFER_MAP
	if (!initMMap())
		return false;
#endif

	return true;
}


bool CameraInfo::initCuda()
{
	// check unified memory support
 	if (cuda_zero_copy){
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		if (!devProp.managedMemory){
			LOG_ERROR("CUDA device does not support managed memory");
			cuda_zero_copy = false;
		}
	}

	// allocate output buffer
	size_t size = frameSize.width * frameSize.height * 3;
	if (cuda_zero_copy)
		cudaMallocManaged(&cuda_out_buffer, size, cudaMemAttachGlobal);
	else
		cuda_out_buffer = (uchar*) malloc(size);
	
	cudaDeviceSynchronize();

	return true;
}


void CameraInfo::deinitCuda()
{
	if (cuda_zero_copy)
		cudaFree(cuda_out_buffer);
	else
		free(cuda_out_buffer);
}

#ifdef BUFFER_MAP
bool CameraInfo::initMMap()
{
	v4l2_requestbuffers req;
	std::memset(&req, 0, sizeof(v4l2_requestbuffers));

	req.count = MMAP_BUFFERS_COUNT;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0){
		LOG_ERROR("Camera V4L2 - failed does not support mmap (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}

 	buffers.resize(req.count);
	
	if (req.count < 2){
		LOG_ERROR("CameraV4L2 - insufficient mmap memory");
		assert(0);
		return false;
	}
	
	v4l2_buffer buff;
	for(size_t n = 0; n < buffers.size(); ++n){
		
		std::memset(&buff, 0, sizeof(v4l2_buffer));
		buff.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buff.memory = V4L2_MEMORY_MMAP;
		buff.index = n;
		
		if (xioctl(fd, VIDIOC_QUERYBUF, &buff) < 0){
			LOG_ERROR("CameraV4L2 -- failed retrieve mmap buffer info (errno=%i) (%s)", errno, strerror(errno));
			assert(0);
			return false;
		}


		buffers[n].length = buff.length;
		buffers[n].start = mmap(nullptr, buff.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buff.m.offset);

		if (buffers[n].start == MAP_FAILED){
			LOG_ERROR("CameraV4L2 -- failed to mmap buffer (errno=%i) (%s)", errno, strerror(errno));
			assert(0);
			return false;
		}
	}

	LOG_DEBUG("CameraV4L2 -- mapped %zu capture buffers with mmap", buffers.size());

	return true;
}


void CameraInfo::deinitMMap()
{
	stopStream();

	int res;
	for (auto& b : buffers){

		if (b.start && (res = munmap(b.start, b.length)) != 0)
			LOG_ERROR("Unmap failed: %d", res);
		b.start = nullptr;
		b.length = 0;
	}
	buffers.clear();
}

#endif

bool CameraInfo::startStream()
{
	if (streamStarted)
		return false;

	v4l2_buffer buff;
	for(size_t i = 0; i < buffers.size(); ++i){
		std::memset(&buff, 0, sizeof(buff));
		buff.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buff.memory = V4L2_MEMORY_MMAP;
		buff.index = i;
		if (xioctl(fd, VIDIOC_QBUF, &buff) == -1){
			LOG_ERROR("VIDIOC_QBUF");
			assert(0);
			return false;
		}

	}

	LOG_DEBUG("CameraV4L2 -- %s starting stream", devicePath.c_str());
	v4l2_buf_type _type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	
	if (xioctl(fd, VIDIOC_STREAMON, &_type) < 0){
		LOG_ERROR("CameraV4L2 -- failed to start streaming (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}

	streamStarted = true;

	return true;
}

bool CameraInfo::stopStream()
{
	if (!streamStarted)
		return true;

	LOG_DEBUG("CameraV4L2 -- %s stopping stream", devicePath.c_str());
	v4l2_buf_type _type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (xioctl(fd, VIDIOC_STREAMOFF, &_type) < 0){
		LOG_ERROR("CameraV4L2 -- failed stop streaming (error=%i) (%s)", errno, strerror(errno));
		assert(0);
	}
	streamStarted = false;
	return true;
}


bool CameraInfo::capture(size_t timeout, cv::Mat& res) const
{
	if (!streamStarted){
		LOG_WARNING("Calling capture while stream is not running");
		return false;
	}
	
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(fd, &fds);
	
	timeval tv;
	tv.tv_sec = 0;
	tv.tv_usec = 0;
	if (timeout > 0){
		tv.tv_sec = timeout / 1000;
		tv.tv_usec = (timeout - (tv.tv_sec * 1000)) * 1000;
	}

	const int result = select(fd + 1, &fds, NULL, NULL, &tv);

	if (result == -1){
		LOG_ERROR("CameraV4L2 -- select() failed (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}
	else if (result == 0){
		if (timeout > 0)
			LOG_ERROR("CameraV4L2 -- select() timed out...");
		return false;
	}
	v4l2_buffer buff;
	std::memset(&buff, 0, sizeof(buff));

	buff.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buff.memory = V4L2_MEMORY_MMAP;
	
	if (xioctl(fd, VIDIOC_DQBUF, &buff) < 0){
		LOG_ERROR("CameraV4L2 -- ioctl(VIDIOC_DQBUF) failed (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}
	
	if (buff.index < 0 || buff.index >= buffers.size()){
		LOG_ERROR("CameraV4L2 - invalid mmap buffer (%u)", buff.index);
		assert(0);
		return false;
	}
	auto tempMat = cv::Mat(frameSize, CV_8UC2, buffers[buff.index].start);
	assert(frameSizeBytes <= tempMat.dataend - tempMat.datastart);
	
	cv::cvtColor(tempMat, res, cv::COLOR_YUV2BGR_UYVY);
	
	if (xioctl(fd, VIDIOC_QBUF, &buff) < 0){
		LOG_ERROR("CameraV4L2 -- ioctl(VIDIOC_QBUF) failed (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}

	return true;
}

// ----------------------END SECTION------------------------


// ------------------------SECTION--------------------------
// ----------InternalCameraParams Implementation------------

bool InternalCameraParams::read(const std::string& filepath, const int num, const cv::Size& resol, const cv::Size& camResol)
{
	std::ifstream ifstrK{filepath + std::to_string(num) + ".K"};
	std::ifstream ifstrDist{filepath + std::to_string(num) + ".dist"};
	
	if (!ifstrK.is_open() || !ifstrDist.is_open()){
		LOG_ERROR("Can't opened file with internal camera params");
		return false;
	}
	
	for(size_t i = 0; i<9; i++)
		ifstrK >> K[i];
	for(size_t j = 0; j<14; ++j)
		ifstrDist >> distortion[j];
	
	captureResolution = camResol;
	resolution = resol;
	ifstrK.close();
	ifstrDist.close();
}

// ----------------------END SECTION------------------------


// ------------------------SECTION--------------------------
// -----------SyncedCameraSource Implementation-------------


int SyncedCameraSource::init(const std::string& param_filepath, const cv::Size& calibSize, const cv::Size& undistSize, const bool useUndist)
{
	bool camsOpenOk = true;
	for (auto& cam : _cams){
		LOG_DEBUG("Initing camera %s...", cam.devicePath.c_str());
		const auto res = cam.init();
		LOG_DEBUG("Initing camera %s %s", cam.devicePath.c_str(), res ? "OK" : "FAILED");
		camsOpenOk |= res;
	}

	if (!camsOpenOk)
		return -1;
	


	if (cudaStreamCreate(&_cudaStream) != cudaError::cudaSuccess){
		_cudaStream = NULL;
		LOG_ERROR("SyncedCameraSource: Failed to create cuda stream");
	}

	size_t planeSize = undistSize.width * undistSize.height * sizeof(uchar);
	for (auto i = 0; i < _cams.size(); ++i){
	    cudaMalloc(&d_src[i], planeSize * 2);
	}

	for(size_t i = 0; i < buffs.size(); ++i){
	      auto& buff = buffs[i];
	      std::memset(&buff, 0, sizeof(v4l2_buffer));
	      buff.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	      buff.memory = V4L2_MEMORY_MMAP;
	}

	if (_undistort){
	      for (size_t i = 0; i < _cams.size(); ++i){
		    if (param_filepath.empty()){
			    LOG_ERROR("Invalid input path with parameter...");
			    return -1;
		    }
		    camIparams[i].read(param_filepath, i, calibSize, frameSize);
		    cv::Mat K(3, 3, CV_64FC1);
		    for (size_t k = 0; k < camIparams[i].K.size(); ++k)
			    K.at<double>(k) = camIparams[i].K[k];
		    cv::Mat D(camIparams[i].distortion);
		    const cv::Size calibratedFrameSize(camIparams[i].resolution);
		    auto& uData = undistFrames[i];
		    cv::Mat newK;
		    if (useUndist)
			    newK = cv::getOptimalNewCameraMatrix(K, D, undistSize, 1,  undistSize, &uData.roiFrame); // 0.0 ? 1.0
		    else
			    newK = cv::getOptimalNewCameraMatrix(K, D, calibratedFrameSize, 1,  undistSize, &uData.roiFrame); // 0.0 ? 1.0
		    Ks[i] = newK;
		    cv::Mat mapX, mapY;
		    cv::initUndistortRectifyMap(K, D, cv::Mat(), newK, undistSize, CV_32FC1, mapX, mapY);
		    uData.remapX.upload(mapX);
		    uData.remapY.upload(mapY);
		    LOG_DEBUG("Generating undistort maps for camera - %i ... OK", i);
	      }
	}

	return 0;
}


bool SyncedCameraSource::startStream()
{
	bool res = true;
	for (auto& c : _cams)
		res |= c.startStream();
	return res;
}

bool SyncedCameraSource::stopStream()
{
	bool res = true;
	for (auto& c : _cams)
		res |= c.stopStream();
	return res;
}


bool SyncedCameraSource::capture(std::array<Frame, 4>& frames)
{
	fd_set fds;
	FD_ZERO(&fds);
	int maxFd = -1;
	for (const auto& c : _cams){
		FD_SET(c.fd, &fds);
		maxFd = std::max(maxFd, c.fd);
	}

	timeval tv;
	tv.tv_sec = 1;
	tv.tv_usec = 0;

	const int result = select(maxFd + 1, &fds, NULL, NULL, &tv);
	
	if (result == -1){
		LOG_ERROR("select() failed (errno=%i) (%s)", errno, strerror(errno));
		assert(0);
		return false;
	}
	else if(result == 0){
		LOG_ERROR("select() timed out...");
		return false;	
	}


	// reading data
	for(const auto& c : _cams){
		if (!FD_ISSET(c.fd, &fds)){
			LOG_DEBUG("Fd %d was not set! Not all cameras ready", c.fd);
			return false;
		}
	}
		

	// dequeue buffers all cameras

	for(size_t i = 0; i < _cams.size(); ++i){
	      auto& buff = buffs[i];
	      const auto& c = _cams[i];
	      auto fd = c.fd;
	      if (xioctl(fd, VIDIOC_DQBUF, &buff) < 0){
		      LOG_ERROR("ioctl(VIDIOC_DQBUF) failed (errno=%i) (%s)", errno, strerror(errno));
		      assert(0);
		      return false;
	      }
	      assert(buff.index >= 0 && buff.index < c.buffers.size());
	}
	

	// do processing
#ifndef NO_OMP
	#pragma omp parallel for default(none) shared(frames)
#endif
	for(size_t i = 0; i < _cams.size(); ++i){
		auto& buff = buffs[i];
		auto& dataBuffer = _cams[i].buffers[buff.index];
		auto* cudaBuffer = _cams[i].cuda_out_buffer;

		gpuConvertUYVY2RGB_opt((uchar*)dataBuffer.start, d_src[i], cudaBuffer, frameSize.width, frameSize.height, _cudaStream);

		const auto uData = cv::cuda::GpuMat(frameSize, CV_8UC3, cudaBuffer);

		if (_undistort){
			cv::cuda::remap(uData, undistFrames[i].undistFrame, undistFrames[i].remapX, undistFrames[i].remapY,
					cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), cudaStreamObj);
			frames[i].gpuFrame = undistFrames[i].undistFrame(undistFrames[i].roiFrame);
		}	
	}


	// enqueue buffer after processing
	for(size_t i = 0; i < _cams.size(); ++i){
		auto& buff = buffs[i];
		const auto& c = _cams[i];
		auto fd = c.fd;
		if (xioctl(fd, VIDIOC_QBUF, &buff) < 0){
			LOG_ERROR("ioctl(VIDIOC_QBUF) failed (errno=%i) (%s)", errno, strerror(errno));
			assert(0);
			return false;
		}
	}

#ifdef NO_COMPILE
	cudaStreamObj.waitForCompletion();
	if (_cudaStream)
		cudaStreamSynchronize(_cudaStream);

#endif
	return true;
}


bool SyncedCameraSource::setFrameSize(const cv::Size& size)
{
	frameSize = size;
	for(auto& c : _cams){
		c.stopStream();
		c.deinit();
		c.frameSize = size;
	}
	return true;
}




// ----------------------END SECTION------------------------





