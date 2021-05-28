#include "view.hpp"
#include "Bowl.hpp"
#include <opencv2/core/opengl.hpp>

#include <Model.hpp>

#include <cuda_gl_interop.h>

void SVView::render(const Camera& cam, const cv::cuda::GpuMat& frame)
{
    // render command
    // ...
    glEnable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer); // bind scene framebuffer
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawSurroundView(cam, frame);

    drawModel(cam);

    glBindFramebuffer(GL_FRAMEBUFFER, 0); // bind default framebuffer
    glDisable(GL_DEPTH_TEST);

    drawQuad(cam);

    // unbound
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


bool SVView::init(const int32 tex_width, const int32 tex_height, const float aspect_ratio_)
{
    if (isInit)
            return isInit;

    /* width and height texture frame from gpuMat */
    this->tex_width = tex_width;
    this->tex_height = tex_height;

    aspect_ratio = aspect_ratio_;

    isInit = initBowl();
    if (!isInit)
      return false;

    isInit = initQuad();
    if (!isInit)
      return false;


    return isInit;
}


void SVView::texturePrepare(const cv::cuda::GpuMat& frame)
{
    if (!texReady){
        texture.create(frame.size(), cv::ogl::Texture2D::Format::RGB, false);
        glBindTexture(GL_TEXTURE_2D, 0);
        texReady = true;
    }

    texture.copyFrom(frame);
}



void SVView::drawSurroundView(const Camera& cam, const cv::cuda::GpuMat& frame)
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

void SVView::drawModel(const Camera& cam)
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


void SVView::drawQuad(const Camera& cam)
{
    frambuffshader.useProgramm();


    glBindVertexArray(quadVAO);
    glBindTexture(GL_TEXTURE_2D, framebuffer_tex);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}


bool SVView::addModel(const std::string& pathmodel, const std::string& pathvertshader,
              const std::string& pathfragshader, const glm::mat4& mat_transform)
{
    bool res = pathmodel.empty() || pathvertshader.empty() || pathfragshader.empty();
    if (res){
      std::cerr << "Error: empty path to model\n";
      return false;
    }

    Model m(pathmodel);
    res = m.getModelInit();
    if (!res){
      std::cerr << "Error: fail load model from path\n";
      return false;
    }


    modelshaders.emplace_back(std::make_shared<Shader>());
    auto last_idx = modelshaders.size() - 1;
    res = modelshaders[last_idx]->initShader(pathvertshader.c_str(), pathfragshader.c_str());
    if (!res){
      std::cerr << "Error: fail init shaders for load model\n";
      return false;
    }


    models.emplace_back(std::move(m));
    modeltranformations.emplace_back(mat_transform);

    return true;
}



bool SVView::initBowl()
{
    bool isgen = SVshader.initShader("shaders/svvert.glsl", "shaders/svfrag.glsl");

    if (!isgen)
        return false;

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
    isgen = bowl.generate_mesh_uv_hole(interpolated_vertices_num, hole_radius, data, idxs);

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

    return true;
}


bool SVView::initQuad()
{
    auto isgen = frambuffshader.initShader("shaders/frame_screenvert.glsl", "shaders/frame_screenfrag.glsl");

    if (!isgen)
        return false;

    constexpr float quadvert[] = {
        -1.f, 1.f, 0.f, 1.f,
        -1.f, -1.f, 0.f, 0.f,
        1.f, 1.f, 1.f, 1.f,
        1.f, -1.f, 1.f, 0.f
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadvert), &quadvert, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2 * sizeof(float)));

    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glGenTextures(1, &framebuffer_tex);
    glBindTexture(GL_TEXTURE_2D, framebuffer_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, wnd_width, wnd_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebuffer_tex, 0);


    glGenRenderbuffers(1, &renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, wnd_width, wnd_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);



    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

