#pragma once
#include <glm.hpp>
#include "SVCudaOGL.hpp"



#include "Virtcam.hpp"
#include "Model.hpp"


#include <vector>

/* data for Reinhard algorithm tonemapping */
struct TonemapConfig
{
    friend class SVRender;
public:
    float intensity;
    float color_adapt;
    float light_adapt;
    TonemapConfig(const float intensity_ = 0.0, const float light_adapt_ = 1.0, const float color_adapt_ = 0.0)
        : intensity(intensity_),light_adapt(light_adapt_), color_adapt(color_adapt_)
    {}

private:
    float t_intensity = 0.0;
    float map_key = 0;
    float gray_mean = 0;
    float gamma = 2.2;
    glm::vec3 chan_mean;
};

class SVRender
{
private:
        bool initBowl();
        bool initQuadRender();
protected:
        void texturePrepare(const cv::cuda::GpuMat& frame);
        void drawSurroundView(const Camera& cam, const cv::cuda::GpuMat& frame);
        void drawModel(const Camera& cam);
        void drawScreen(const Camera& cam);

public:  
       bool getInit() const{return isInit;}
       bool addModel(const std::string& pathmodel, const std::string& pathvertshader,
                     const std::string& pathfragshader, const glm::mat4& mat_transform);
       void contrastCorrectionParameters(const cv::cuda::GpuMat& frame, std::shared_ptr<TonemapConfig>& tmc);
public:
        SVRender(const int32 wnd_width_, const int32 wnd_height_);


        SVRender& operator=(const SVRender&) = delete;
        SVRender(const SVRender&) = delete;
	
        bool init();
        void render(const Camera& cam, const cv::cuda::GpuMat& frame);
        void addTonemappingCfg(std::shared_ptr<TonemapConfig>& tmc_);
private:
        OGLBuffer OGLbowl;
        OGLBuffer OGLquadrender;
        float aspect_ratio;
        int32  wnd_width;
        int32  wnd_height;
        CUDA_OGL cuOgl;
        std::shared_ptr<TonemapConfig> tmc;
private:
        std::vector<Model> models;
        std::vector<std::shared_ptr<Shader>> modelshaders;
        std::vector<glm::mat4> modeltranformations;
        std::vector<cv::cuda::GpuMat> chan_tonemap;
        cv::cuda::GpuMat gray_img, log_img;
        cv::cuda::Stream streamObj;
        bool isInit = false;
        bool texReady;
};






