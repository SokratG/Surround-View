#pragma once
#include <Shader.hpp>

#include <stdint.h>

#include <cuda_runtime.h>

#include <opencv2/core/cuda.hpp>


using uint = uint32_t;
using int32 = int32_t;

struct CUDA_OGL
{
private:
    cudaGraphicsResource_t cuRes;
    GLuint cuGlBuf;
    GLuint idTex;
    GLint isInit;
protected:
    void genBuffer(const GLsizeiptr size_);
    void genTexture(int width, int height);
    void clear();

    /* not implemented */
    bool copyTo(cv::cuda::GpuMat& frame, cudaStream_t cuStream = 0);
public:
    CUDA_OGL() : cuRes(0), cuGlBuf(0), idTex(0), isInit(0) {}
    ~CUDA_OGL();
    bool init(const cv::cuda::GpuMat& frame);
    bool copyFrom(const cv::cuda::GpuMat& frame, const uint tex_id, cudaStream_t cuStream = 0);

};


struct OGLBuffer
{
    GLuint VAO;
    GLuint VBO;
    GLuint EBO;
    uint indexBuffer;
    Shader OGLShader;
    GLuint framebuffer;
    GLuint renderbuffer;
    GLuint framebuffer_tex;
    OGLBuffer();
    ~OGLBuffer();

    OGLBuffer& operator=(const OGLBuffer&) = delete;
    OGLBuffer(const OGLBuffer&) = delete;
private:
    void clearBuffers();

};
