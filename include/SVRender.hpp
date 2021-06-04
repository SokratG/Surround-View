#pragma once
#include <glm.hpp>
#include "SVCudaOGL.hpp"

#include <opencv2/core/opengl.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "Virtcam.hpp"
#include "Model.hpp"


#include <vector>


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

public:
        SVRender(const int32 wnd_width_, const int32 wnd_height_);


        SVRender& operator=(const SVRender&) = delete;
        SVRender(const SVRender&) = delete;
	
        bool init();
        void render(const Camera& cam, const cv::cuda::GpuMat& frame);
private:
        OGLBuffer OGLbowl;
        OGLBuffer OGLquadrender;
        float aspect_ratio;
        int32  wnd_width;
        int32  wnd_height;
        CUDA_OGL cuOgl;
private:
        std::vector<Model> models;
        std::vector<std::shared_ptr<Shader>> modelshaders;
        std::vector<glm::mat4> modeltranformations;
        bool isInit = false;
        bool texReady;
};






