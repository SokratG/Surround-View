#include <SVDisplay.hpp>


extern bool finish; // from SVApp

float lastX = 1280 / 2.f; // last x pos cursor
float lastY = 720 / 2.f; // last y pos cursor
float deltaTime = 0.f;
float lastFrame = 0.f;
bool firstMouse = true;
bool demo_key_press = false;
bool topview_key_press = false;


static void frame_buffer_size_callback(GLFWwindow* wnd, int width, int height)
{
        glViewport(0, 0, width, height);
}

static void processMouse(GLFWwindow* window, double xpos, double ypos)
{
        if (firstMouse)
        {
                lastX = xpos;
                lastY = ypos;
                firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed: y ranges bottom to top
        lastX = xpos;
        lastY = ypos;

        cam.processMouseMovement(xoffset, yoffset);
}

static void processScroll(GLFWwindow* window, double xoffset, double yoffset)
{
        cam.processMouseScroll(yoffset);
}


static void processInput(GLFWwindow* window, SVDisplayView* svdisp)
{
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
        }

        const float cameraSpeed = 1.0f * deltaTime; //2.5f
        constexpr auto const_speed = 0.5f;

        if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS && !demo_key_press) {
                svdisp->setTVMode(false);
                svdisp->setSVMode(!svdisp->getSVMode());
                demo_key_press = true;
        }
        if (glfwGetKey(window, GLFW_KEY_V) == GLFW_RELEASE) {
                demo_key_press = false;
        }

        if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && !topview_key_press) {
                svdisp->setSVMode(false);
                svdisp->setTVMode(!svdisp->getTVMode());
                topview_key_press = true;
        }
        if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE) {
                topview_key_press = false;
        }

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                finish = true;
        }

        /* reset camera position */
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
                svdisp->setSVMode(false);
                svdisp->setTVMode(false);
                svdisp->resetCameraState();
        }

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                cam.processKeyboard(Camera_Movement::FORWARD, const_speed);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                cam.processKeyboard(Camera_Movement::BACKWARD, const_speed);
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                cam.processKeyboard(Camera_Movement::LEFT, const_speed);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                cam.processKeyboard(Camera_Movement::RIGHT, const_speed);
        }

}



bool SVDisplayView::init(const int32 wnd_width, const int32 wnd_height, std::shared_ptr<SVRender> scene_view)
{
        if (isInit)
                return isInit;

        this->width = wnd_width;
        this->height = wnd_height;
        aspect_ratio = static_cast<float>(wnd_width) / wnd_height;
        disp_view = scene_view;

        /*
            glfw initialize
        */
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);


        window = glfwCreateWindow(width, height, "Surround View", NULL, NULL);

        if (window == nullptr) {
            std::cerr << "Failed create GLFW window\n";
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);



        glEnable(GL_DEPTH_TEST);
        //glDepthFunc(GL_LEQUAL);

        glViewport(0, 0, width, height);

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        // reszie callback
        glfwSetFramebufferSizeCallback(window, frame_buffer_size_callback);
        // set mouse pos callback
        glfwSetCursorPosCallback(window, processMouse);
        // set mouse scroll callback
        glfwSetScrollCallback(window, processScroll);


        isInit = true;
        return isInit;
}

bool SVDisplayView::render(const cv::cuda::GpuMat& frame)
{
    if (!glfwWindowShouldClose(window) && isInit) {
        // input
        processInput(window, this);

        if (useDemoSurroundView)
            demoSVMode(cam);
        if (useDemoTopView)
            demoTVMode(cam);

        if (disp_view->getInit()){
            disp_view->render(cam, frame);
        }

        // check and call events and swap the buffers
        glfwPollEvents();
        glfwSwapBuffers(window);


        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
    }
    else
        return false;

    return true;
}



void SVDisplayView::resetCameraState()
{
    cam = Camera(glm::vec3(0.0, 1.7, 1.0), glm::vec3(0.0, 1.0, 0.0));
}

void SVDisplayView::demoSVMode(Camera& camera)
{
    constexpr auto const_speed = 0.03;
    camera.processKeyboard(Camera_Movement::LEFT, const_speed);
    constexpr auto xoffset = 15.0;
    camera.processMouseMovement(xoffset, 0);
}

void SVDisplayView::demoTVMode(Camera& camera)
{
    constexpr auto angle = -90.0;
    constexpr auto rot_angle_x = 7.5;
    camera.processMouseMovement(0, angle);
    camera.processMouseMovement(rot_angle_x, 0);
}



void SVDisplayView::setSVMode(const bool demo)
{
    useDemoSurroundView = demo;
    resetCameraState();
    cam.setCamPos(glm::vec3(0, 2.5, 2.75));
    cam.processMouseMovement(0, -315);
}

bool SVDisplayView::getSVMode() const
{
    return useDemoSurroundView;
}

void SVDisplayView::setTVMode(const bool topview)
{
    useDemoTopView = topview;
    resetCameraState();
    cam.setCamPos(glm::vec3(0, 4.15, 0.0));
}

bool SVDisplayView::getTVMode() const
{
    return useDemoTopView;
}

