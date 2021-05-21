#pragma once
#include <memory>
#include <stdint.h>
#include "view.hpp"
#define GLFW_INCLUDE_ES32
#include "glfw3/glfw3.h"


using int32 = int32_t;
Camera cam(glm::vec3(0.f, 1.f, 1.f), glm::vec3(0.f, 1.f, 0.f));

float lastX = 1280 / 2.f; // last x pos cursor
float lastY = 720 / 2.f; // last y pos cursor
float deltaTime = 0.f;
float lastFrame = 0.f;
bool firstMouse = true;

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


static void processInput(GLFWwindow* window)
{
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
        }

        const float cameraSpeed = 1.0f * deltaTime; //2.5f
        constexpr auto const_speed = 0.5f;

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


class DisplayView
{
private:
	GLFWwindow* window;
        int32 width, height;
        float aspect_ratio;
        std::shared_ptr<View> disp_view;
        bool isInit;
public:
        DisplayView() : window(nullptr), disp_view(nullptr){
            width = 0;
            height = 0;
            isInit = false;
	}
        ~DisplayView(){glfwTerminate();}

        bool init(const int32 wnd_width, const int32 wnd_height,
                  const int32 tex_width, const int32 tex_height, std::shared_ptr<View> scene_view){
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
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2); // 3
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);


            window = glfwCreateWindow(width, height, "Surround View", NULL, NULL);

            if (window == nullptr) {
                std::cerr << "Failed create GLFW window\n";
                glfwTerminate();
                return false;
            }
            glfwMakeContextCurrent(window);

            if (!disp_view->getInit())
                disp_view->init(tex_width, tex_height, aspect_ratio);

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

        bool render(const cv::cuda::GpuMat& frame)
        {

            if (!glfwWindowShouldClose(window) && isInit) {
                // input
                processInput(window);


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

};




