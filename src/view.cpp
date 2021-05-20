#include "view.hpp"
#include "Bowl.hpp"
#include <opencv2/core/opengl.hpp>

#include <Model.hpp>

#include <cuda_gl_interop.h>

static void renderQuad();

void View::render(const Camera& cam, const cv::cuda::GpuMat& frame)
{
    // render command
    // ...
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    drawSurroundView(cam, frame);

    drawModel(cam);

    glDisable(GL_BLEND);

    // unbound
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


bool View::init(const int32 width, const int32 height, const float aspect_ratio_)
{
    if (isInit)
            return isInit;

    /* width and height texture frame from gpuMat */
    this->width = width;
    this->height = height;

    aspect_ratio = aspect_ratio_;

    glBindFramebuffer(GL_FRAMEBUFFER, 0); // bind default framebuffer

    SVshader.initShader("shaders/svvert.glsl", "shaders/svfrag.glsl");

    glGenVertexArrays(1, &bowlVAO);
    glGenBuffers(1, &bowlVBO);
    glGenBuffers(1, &bowlEBO);


    /* Bowl parameter */
    constexpr auto inner_radius = 0.3f;
    constexpr auto radius = 0.4f;
    constexpr auto hole_radius = 0.07f;
    constexpr auto interpolated_vertices_num = 750.f;
    constexpr auto a = 0.4f;
    constexpr auto b = 0.4f;
    constexpr auto c = 0.15f;

    Bowl bowl(inner_radius, radius, a, b, c);
    std::vector<float> data;
    std::vector<uint> idxs;
    bool isgen = bowl.generate_mesh_uv_hole(interpolated_vertices_num, hole_radius, data, idxs);

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



void View::drawSurroundView(const Camera& cam, const cv::cuda::GpuMat& frame)
{
    glm::mat4 model(1.f);
    auto view = cam.getView();
    auto projection = glm::perspective(glm::radians(cam.getCamZoom()), aspect_ratio, 0.1f, 100.f);


    model = glm::scale(model, glm::vec3(5.f, 5.f, 5.f));


    SVshader.useProgramm();
    SVshader.setMat4("model", model);
    SVshader.setMat4("view", view);
    SVshader.setMat4("projection", projection);
    texturePrepare(frame);

    glBindVertexArray(bowlVAO);

    texture.bind();   

    glDrawElements(GL_TRIANGLE_STRIP, indexPartBowl, GL_UNSIGNED_INT, 0);
}

void View::drawModel(const Camera& cam)
{
    glm::mat4 model(1.f);
    auto view = cam.getView();
    auto projection = glm::perspective(glm::radians(cam.getCamZoom()), aspect_ratio, 0.1f, 100.f);

    for(auto i = 0; i < models.size(); ++i){
        model = modeltranformations[i];
        modelshaders[i]->useProgramm();
        modelshaders[i]->setMat4("model", model);
        modelshaders[i]->setMat4("view", view);
        modelshaders[i]->setMat4("projection", projection);
        models[i].Draw(*modelshaders[i]);
    }

}


bool View::addModel(const std::string& pathmodel, const std::string& pathvertshader,
              const std::string& pathfragshader, const glm::mat4& mat_transform)
{
    bool res = pathmodel.empty() || pathvertshader.empty() || pathfragshader.empty();
    if (res)
      return false;

    Model m(pathmodel);
    res = m.getModelInit();
    if (!res)
      return false;


    modelshaders.emplace_back(std::make_shared<Shader>());
    auto last_idx = modelshaders.size() - 1;
    res = modelshaders[last_idx]->initShader(pathvertshader.c_str(), pathfragshader.c_str());
    if (!res)
      return false;


    models.emplace_back(std::move(m));
    modeltranformations.emplace_back(mat_transform);

    return true;
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

