#include <SVRender.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cuda_gl_interop.h>


SVRender::SVRender(const int32 wnd_width_, const int32 wnd_height_) :
    wnd_width(wnd_width_), wnd_height(wnd_height_), aspect_ratio(0.f), texReady(false),
    tonemap_luminance(1.0)
{

}



void SVRender::render(const Camera& cam, const cv::cuda::GpuMat& frame)
{
    // render command
    // ...
    glEnable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, OGLquadrender.framebuffer); // bind scene framebuffer
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawSurroundView(cam, frame);

    drawModel(cam);

    glBindFramebuffer(GL_FRAMEBUFFER, 0); // bind default framebuffer
    glDisable(GL_DEPTH_TEST);

    drawScreen(cam);

    // unbound
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


bool SVRender::init(const ConfigBowl& cbowl, const std::string& shadersurroundvert, const std::string& shadersurroundfrag,
                    const std::string& shaderscreenvert, const std::string& shaderscreenfrag)
{
    if (isInit)
            return isInit;

    aspect_ratio = static_cast<float>(wnd_width) / wnd_height;

    isInit = initBowl(cbowl, shadersurroundvert, shadersurroundfrag);
    if (!isInit)
      return false;

    isInit = initQuadRender(shaderscreenvert, shaderscreenfrag);
    if (!isInit)
      return false;

    return isInit;
}


void SVRender::texturePrepare(const cv::cuda::GpuMat& frame)
{
    if (!texReady){
        texReady = cuOgl.init(frame);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    auto ok = cuOgl.copyFrom(frame);
}


void SVRender::drawSurroundView(const Camera& cam, const cv::cuda::GpuMat& frame)
{
    glm::mat4 model = bowlmodel.transformation;
    auto view = cam.getView();
    auto projection = glm::perspective(glm::radians(cam.getCamZoom()), aspect_ratio, 0.1f, 100.f);


    model = glm::scale(model, glm::vec3(5.f, 5.f, 5.f));

    OGLbowl.OGLShader.useProgramm();
    OGLbowl.OGLShader.setMat4("model", model);
    OGLbowl.OGLShader.setMat4("view", view);
    OGLbowl.OGLShader.setMat4("projection", projection);
    OGLbowl.OGLShader.setFloat("lum_white", tonemap_luminance);

    texturePrepare(frame);

    glBindVertexArray(OGLbowl.VAO);

    glDrawElements(GL_TRIANGLE_STRIP, OGLbowl.indexBuffer, GL_UNSIGNED_INT, 0);
}

void SVRender::drawModel(const Camera& cam)
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


void SVRender::drawBlackRect(const Camera& cam)
{

}

void SVRender::drawScreen(const Camera& cam)
{
    OGLquadrender.OGLShader.useProgramm();


    glBindVertexArray(OGLquadrender.VAO);
    glBindTexture(GL_TEXTURE_2D, OGLquadrender.framebuffer_tex);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

}


bool SVRender::addModel(const std::string& pathmodel, const std::string& pathvertshader,
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



bool SVRender::initBowl(const ConfigBowl& cbowl, const std::string& shadersurroundvert, const std::string& shadersurroundfrag)
{
    bool isinit = OGLbowl.OGLShader.initShader(shadersurroundvert.c_str(), shadersurroundfrag.c_str());

    if (!isinit)
        return false;

    glGenVertexArrays(1, &OGLbowl.VAO);
    glGenBuffers(1, &OGLbowl.VBO);
    glGenBuffers(1, &OGLbowl.EBO);

    bowlmodel = cbowl;
    std::vector<float> data;
    std::vector<uint> idxs;

    Bowl bowl(bowlmodel);
    isinit = bowl.generate_mesh_uv_hole(cbowl.vertices_num, cbowl.hole_radius, data, idxs);

    if (!isinit)
        return false;


    OGLbowl.indexBuffer = idxs.size();

    constexpr auto stride = (3 + 2) * sizeof(float);


    glBindVertexArray(OGLbowl.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, OGLbowl.VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, OGLbowl.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, OGLbowl.indexBuffer * sizeof(uint), &idxs[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));

    return true;
}


bool SVRender::initQuadRender(const std::string& shaderscreenvert, const std::string& shaderscreenfrag)
{
    auto isinit = OGLquadrender.OGLShader.initShader(shaderscreenvert.c_str(), shaderscreenfrag.c_str());

    if (!isinit)
        return false;

    constexpr float quadvert[] = {
        -1.f, 1.f, 0.f, 1.f,
        -1.f, -1.f, 0.f, 0.f,
        1.f, 1.f, 1.f, 1.f,
        1.f, -1.f, 1.f, 0.f
    };

    glGenVertexArrays(1, &OGLquadrender.VAO);
    glGenBuffers(1, &OGLquadrender.VBO);
    glBindVertexArray(OGLquadrender.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, OGLquadrender.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadvert), &quadvert, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2 * sizeof(float)));

    glGenFramebuffers(1, &OGLquadrender.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, OGLquadrender.framebuffer);

    glGenTextures(1, &OGLquadrender.framebuffer_tex);
    glBindTexture(GL_TEXTURE_2D, OGLquadrender.framebuffer_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, wnd_width, wnd_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, OGLquadrender.framebuffer_tex, 0);


    glGenRenderbuffers(1, &OGLquadrender.renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, OGLquadrender.renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, wnd_width, wnd_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, OGLquadrender.renderbuffer);



    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return false;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

