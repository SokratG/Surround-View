#include "CudaOGL.hpp"



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


