#pragma once
#include <vector>

#include <Bowl.hpp>
#include <Virtcam.hpp>
#include <Model.hpp>

#include <SVCudaOGL.hpp>


class SVRender
{
private:
        bool initBowl(const ConfigBowl& cbowl, const std::string& filesurroundvert, const std::string& filesurroundfrag);
        bool initQuadRender(const std::string& filescreenvert, const std::string& filescreenfrag);
protected:
        void texturePrepare(const cv::cuda::GpuMat& frame);
        void drawSurroundView(const Camera& cam, const cv::cuda::GpuMat& frame);
        void drawModel(const Camera& cam);
        void drawScreen(const Camera& cam);

public:  
       bool getInit() const{return isInit;}
       bool addModel(const std::string& pathmodel, const std::string& pathvertshader,
                     const std::string& pathfragshader, const glm::mat4& mat_transform);
       float getLuminance() const{return tonemap_luminance;}
       void setLuminance(const float tonemap_luminance_) {tonemap_luminance = tonemap_luminance_;}

public:
        SVRender(const int32 wnd_width_, const int32 wnd_height_);


        SVRender& operator=(const SVRender&) = delete;
        SVRender(const SVRender&) = delete;
	
        bool init(const ConfigBowl& cbowl, const std::string& shadersurroundvert, const std::string& shadersurroundfrag,
                  const std::string& shaderscreenvert, const std::string& shaderscreenfrag);
        void render(const Camera& cam, const cv::cuda::GpuMat& frame);

private:
        ConfigBowl bowlmodel;
        OGLBuffer OGLbowl;
        OGLBuffer OGLquadrender;
        float aspect_ratio;
        int32  wnd_width;
        int32  wnd_height;
        CUDA_OGL cuOgl;
        float tonemap_luminance;
private:
        std::vector<Model> models;
        std::vector<std::shared_ptr<Shader>> modelshaders;
        std::vector<glm::mat4> modeltranformations;
        bool isInit = false;
        bool texReady;
};






