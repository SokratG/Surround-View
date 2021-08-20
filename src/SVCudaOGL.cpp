#include <SVCudaOGL.hpp>

#include <opencv2/core/cuda.hpp>

#include <cuda_gl_interop.h>

#define GL_BGR  0x80E0
#define GL_BGRA 0x80E1

// ------------------------------- CUDA_OGL --------------------------------

CUDA_OGL::~CUDA_OGL()
{
    clear();
}

bool CUDA_OGL::init(const cv::cuda::GpuMat& frame)
{
    const int width_ = frame.cols;
    const int height_ = frame.rows;

    clear();

    genTexture(width_, height_);

    const GLsizeiptr size_ = width_ * height_ * frame.elemSize();

    genBuffer(size_);

    cudaError_t cu_status = cudaGraphicsGLRegisterBuffer(&cuRes, cuGlBuf, cudaGraphicsMapFlagsNone);

    if (cu_status != cudaSuccess)
        return false;

    cu_status = cudaGraphicsMapResources(1, &cuRes);
    if (cu_status != cudaSuccess){
        clear();
        return false;
    }


    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}

void CUDA_OGL::genTexture(int width, int height)
{

    glGenTextures(1, &idTex);

    glBindTexture(GL_TEXTURE_2D, idTex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, (GLvoid*)0);

    glGenerateMipmap(GL_TEXTURE_2D);

}


void CUDA_OGL::genBuffer(const GLsizeiptr size_)
{
    glGenBuffers(1, &cuGlBuf);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cuGlBuf);

    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void CUDA_OGL::clear()
{
    if (cuRes){
      cudaGraphicsUnmapResources(1, &cuRes);
      cudaGraphicsUnregisterResource(cuRes);
    }
    if (idTex)
      glDeleteTextures(1, &idTex);
    if(cuGlBuf)
      glDeleteBuffers(1, &cuGlBuf);

    cuRes = 0;
    cuGlBuf = 0;
    idTex = 0;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, 0);
}



bool CUDA_OGL::copyFrom(const cv::cuda::GpuMat& frame, const uint tex_id, cudaStream_t cuStream)
{

    if (!cuRes || !cuGlBuf || !idTex)
      return false;

    GLenum err = glGetError();


    const int width_ = frame.cols;
    const int height_ = frame.rows;
    const int cu_width_ = width_ * frame.elemSize();
    const int spitch = frame.step;
    const int dpitch = cu_width_;

    void* dst;
    size_t size;
    cudaError_t cu_status = cudaGraphicsResourceGetMappedPointer(&dst, &size, cuRes);

    if (cu_status != cudaSuccess)
        return false;

    if (cuStream == 0)
        cu_status = cudaMemcpy2D(dst, dpitch, frame.data, spitch, cu_width_, height_, cudaMemcpyDeviceToDevice);
    else
        cu_status = cudaMemcpy2DAsync(dst, dpitch, frame.data, spitch, cu_width_, height_, cudaMemcpyDeviceToDevice, cuStream);

    if (cu_status != cudaSuccess)
        return false;


    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cuGlBuf);

    glActiveTexture(GL_TEXTURE0 + tex_id);

    glBindTexture(GL_TEXTURE_2D, idTex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, height_, GL_BGR, GL_UNSIGNED_BYTE, (GLvoid*)0);

    glGenerateMipmap(GL_TEXTURE_2D);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    return true;

}


bool CUDA_OGL::copyTo(cv::cuda::GpuMat& frame, cudaStream_t cuStream)
{
      if (!cuRes || !cuGlBuf || !idTex)
        return false;

      GLenum err = glGetError();

      const int width_ = frame.cols;
      const int height_ = frame.rows;
      const int cu_width_ = width_ * frame.elemSize();
      const int spitch = frame.step;
      const int dpitch = cu_width_;

      // OpenGL ES doesn't support glGetTexImage, but can use glReadPixels(store in host memory, copy again to device memory)

      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

      return false;
}






// ------------------------------- OGLBuffer --------------------------------

OGLBuffer::OGLBuffer() : VAO(0), VBO(0), EBO(0), indexBuffer(0), framebuffer(0), renderbuffer(0), framebuffer_tex(0)
{

}


OGLBuffer::~OGLBuffer()
{
    clearBuffers();
}

void OGLBuffer::clearBuffers()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteRenderbuffers(1, &renderbuffer);
    glDeleteTextures(1, &framebuffer_tex);
}

