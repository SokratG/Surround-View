#include "shader.hpp"

#include <stdint.h>

using uint = uint32_t;
using int32 = int32_t;



struct CUDA_OGL
{

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
