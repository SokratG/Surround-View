#pragma once
#include <glm.hpp>
#include <opencv2/opencv.hpp>
#include <GLES3/gl32.h>
#include <EGL/egl.h>

#include <opencv2/core/opengl.hpp>

#include "virtcam.hpp"
#include "shader.hpp"
#include "Model.hpp"

#include <vector>
#include <stdint.h>

using uint = unsigned int;
using int32 = int32_t;



class View
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
        View(const int32 wnd_width_, const int32 wnd_height_) :
            wnd_width(wnd_width_), wnd_height(wnd_height_), aspect_ratio(0.f), tex_width(0), tex_height(0), texReady(false)
        {}
        ~View(){clearBuffers();}

	View& operator=(const View&) = delete;
	View(const View&) = delete;
	
	void clearBuffers(){
                glDeleteVertexArrays(1, &bowlVAO);
                glDeleteBuffers(1, &bowlVBO);
                glDeleteBuffers(1, &bowlEBO);
	}
	

        bool init(const int32 tex_width, const int32 tex_height, const float aspect_ratio_);
        void render(const Camera& cam, const cv::cuda::GpuMat& frame);
private:
        GLuint bowlVAO {0};
        GLuint bowlVBO {0};
        GLuint bowlEBO {0};
        uint indexPartBowl {0};
        float aspect_ratio;
        int32 tex_width;
        int32 tex_height;
        cv::ogl::Texture2D texture;
        Shader SVshader;
private:
        int32  wnd_width;
        int32  wnd_height;
        Shader frambuffshader;
        GLuint framebuffer{0};
        GLuint renderbuffer{0};
        GLuint framebuffer_tex{0};
        GLuint quadVAO{0};
        GLuint quadVBO{0};
private:
        std::vector<Model> models;
        std::vector<std::shared_ptr<Shader>> modelshaders;
        std::vector<glm::mat4> modeltranformations;
        bool isInit = false;
        bool texReady;
};






