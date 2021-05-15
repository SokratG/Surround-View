#pragma once
#include <glm.hpp>
#include <opencv2/opencv.hpp>
#include <GLES3/gl32.h>
#include <EGL/egl.h>

#include <opencv2/core/opengl.hpp>

#include "virtcam.hpp"
#include "shader.hpp"


#include <vector>
#include <stdint.h>

using uint = unsigned int;
using int32 = int32_t;
using vec3 = glm::vec3;
using point3 = glm::vec3;
using mat4 = glm::mat4;
using mat3 = glm::mat3;


class View
{
private:
        GLuint bowlVAO {0};
        GLuint bowlVBO {0};
        GLuint bowlEBO {0};
        uint indexPartBowl {0};
        float aspect_ratio;
        int32 width;
        int32 height;
        cv::ogl::Texture2D texture;
        std::shared_ptr<Camera> cam;
        Shader SVshader;
        Shader modelShader;
        bool isInit = false;
        bool texReady;

protected:
        void texturePrepare(const cv::cuda::GpuMat& frame);
        void drawSurroundView(const Camera& cam, const cv::cuda::GpuMat& frame);
        void drawModel(const Camera& cam);
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

public:
        View() :
            aspect_ratio(0.f), width(0), height(0), texReady(false)
        {}
        ~View(){clearBuffers();}

	View& operator=(const View&) = delete;
	View(const View&) = delete;
	
	void clearBuffers(){
                glDeleteVertexArrays(1, &bowlVAO);
                glDeleteBuffers(1, &bowlVBO);
                glDeleteBuffers(1, &bowlEBO);
	}
	

        bool init(const int32 width, const int32 height);
        void render(const Camera& cam, const cv::cuda::GpuMat& frame);
};






