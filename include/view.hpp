#pragma once
#include <glm.hpp>
#include "CudaOGL.hpp"

#include <opencv2/core/opengl.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "virtcam.hpp"
#include "Model.hpp"


#include <vector>




class SVRender
{
private:
        bool initBowl();
        bool initQuad();
protected:
        void texturePrepare(const cv::cuda::GpuMat& frame);
        void drawSurroundView(const Camera& cam, const cv::cuda::GpuMat& frame);
        void drawModel(const Camera& cam);
        void drawQuad(const Camera& cam);
#ifdef NO
        void glew_init(){
            glewExperimental = GL_TRUE;
            GLenum err = glewInit();
            if  (GLEW_OK != err){
                std::cerr << "Error: " << glewGetErrorString(err) << "\n";
                exit(EXIT_FAILURE);
            }
        }
#endif
public:  
       bool getInit() const{return isInit;}
       bool addModel(const std::string& pathmodel, const std::string& pathvertshader,
                     const std::string& pathfragshader, const glm::mat4& mat_transform);

public:
        SVRender(const int32 wnd_width_, const int32 wnd_height_) :
            wnd_width(wnd_width_), wnd_height(wnd_height_), aspect_ratio(0.f), tex_width(0), tex_height(0), texReady(false)
        {}

        SVRender& operator=(const SVRender&) = delete;
        SVRender(const SVRender&) = delete;
	
        bool init(const int32 tex_width, const int32 tex_height, const float aspect_ratio_);
        void render(const Camera& cam, const cv::cuda::GpuMat& frame);
private:
        OGLBuffer OGLbowl;
        OGLBuffer OGLquadrender;
        float aspect_ratio;
        int32 tex_width;
        int32 tex_height; 
        int32  wnd_width;
        int32  wnd_height;
        cv::ogl::Texture2D texture;
private:
        std::vector<Model> models;
        std::vector<std::shared_ptr<Shader>> modelshaders;
        std::vector<glm::mat4> modeltranformations;
        bool isInit = false;
        bool texReady;
};






