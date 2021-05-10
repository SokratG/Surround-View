#include "view.hpp"
#include "Bowl.hpp"
#include <opencv2/core/opengl.hpp>

#include <cuda_gl_interop.h>

static void renderQuad();

constexpr auto bolw_size = 3.f * 3.14159265359f / 2.f;

void View::render(const Camera& cam, const cv::cuda::GpuMat& frame)
{
    // render command
    // ...
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glm::mat4 model(1.f);
    //model = glm::scale(model, glm::vec3(2.f, 2.f, 2.f));
    auto view = cam.getView();
    auto projection = glm::perspective(glm::radians(cam.getCamZoom()), aspect_ratio, 0.1f, 100.f);

    SVshader.useProgramm();
    SVshader.setMat4("model", model);
    SVshader.setMat4("view", view);
    SVshader.setMat4("projection", projection);
    texturePrepare(frame);

    glBindVertexArray(bowlVAO);

    texture.bind();

    glDrawElements(GL_TRIANGLE_STRIP, indexPartBowl, GL_UNSIGNED_INT, 0);

    // unbound
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


bool View::init(const int32 width, const int32 height)
{
    if (isInit)
            return isInit;


    this->width = width;
    this->height = height;
    aspect_ratio = static_cast<float>(width) / height;

    glBindFramebuffer(GL_FRAMEBUFFER, 0); // bind default framebuffer
    SVshader.initShader("shaders/svvert.glsl", "shaders/svfrag.glsl");

    glGenVertexArrays(1, &bowlVAO);
    glGenBuffers(1, &bowlVBO);
    glGenBuffers(1, &bowlEBO);

    auto inner_radius = 0.3f;
    auto radius = 0.4f;
    auto a = 0.4f;
    auto b = 0.4f;
    auto c = 0.1f;

    Bowl bowl(inner_radius, radius, a, b, c);
    std::vector<float> data;
    std::vector<uint> idxs;
    //bool isgen = bowl.generate_mesh_uv(80.f, data, idxs);
    bool isgen = bowl.generate_mesh_uv_hole(60.f, 0.02f, data, idxs);
    if (!isgen)
        return false;


    indexPartBowl= idxs.size();

    constexpr auto stride = (3 + 2) * sizeof(float);


    glBindVertexArray(bowlVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bowlVBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bowlEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexPartBowl * sizeof(uint), &idxs[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));


    isInit = true;
    return isInit;
}


void View::texturePrepare(const cv::cuda::GpuMat& frame)
{
    if (!texReady){
        texture.create(frame.size(), cv::ogl::Texture2D::Format::RGB, false);
        glBindTexture(GL_TEXTURE_2D, 0);
        texReady = true;
    }

    texture.copyFrom(frame);
}









static uint quadVao = 0;
static uint quadVbo;
void renderQuad()
{
    if (quadVao == 0){
        float quadvert[] = {
            -1.f, 1.f, 0.f, 0.f, 1.f,
            -1.f, -1.f, 0.f, 0.f, 0.f,
            1.f, 1.f, 0.f, 1.f, 1.f,
            1.f, -1.f, 0.f, 1.f, 0.f
        };
        glGenVertexArrays(1, &quadVao);
        glGenBuffers(1, &quadVbo);
        glBindVertexArray(quadVao);
        glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadvert), &quadvert, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

