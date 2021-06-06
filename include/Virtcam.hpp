#pragma once
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


// Defines several possible options for camera movement
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

// Default camera values
using DefaultCameraOpt = struct DefaultCameraOpt_
{
    static constexpr float YAW = -90.0f;
    static constexpr float PITCH = 0.0f;
    static constexpr float SPEED = 2.5f;
    static constexpr float SENSITIVITY = 0.1f;
    static constexpr float ZOOM = 45.0f;
};



class Camera
{
public:

    Camera(glm::vec3 position = glm::vec3(0.f, 0.f, 0.f), glm::vec3 up = glm::vec3(0.f, 1.f, 0.f), float yaw = DefaultCameraOpt::YAW, float pitch = DefaultCameraOpt::PITCH) :
        camFront(glm::vec3(0.f, 0.f, -1.f)), mvspeed(DefaultCameraOpt::SPEED), sensitivity(DefaultCameraOpt::SENSITIVITY), zoom(DefaultCameraOpt::ZOOM)
    {
        camPosition = position;
        worldUp = up;
        this->yaw = yaw;
        this->pitch = pitch;
        updateCameraVectors();
    }
    // constructor with scalar values
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) :
         camFront(glm::vec3(0.0f, 0.0f, -1.0f)), mvspeed(DefaultCameraOpt::SPEED), sensitivity(DefaultCameraOpt::SENSITIVITY), zoom(DefaultCameraOpt::ZOOM)
    {
        camPosition = glm::vec3(posX, posY, posZ);
        worldUp = glm::vec3(upX, upY, upZ);
        this->yaw = yaw;
        this->pitch = pitch;
        updateCameraVectors();
    }


    /*
    *  returns the view matrix calculated using Euler Angles and the LookAt Matrix
    */
    glm::mat4 getView() const {
        return glm::lookAt(camPosition, camPosition + camFront, camUp);
    }

    /*
    *  processes input data from handlers keyboard
    */
    void processKeyboard(Camera_Movement direction, float deltaTime)
    {
        float velocity = mvspeed * deltaTime;
        if (direction == FORWARD)
            camPosition += camFront * velocity;
        if (direction == BACKWARD)
            camPosition -= camFront * velocity;
        if (direction == LEFT)
            camPosition -= camRight * velocity;
        if (direction == RIGHT)
            camPosition += camRight * velocity;
    }

    /*
    *  processes input data from handlers mouse moves
    */
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true)
    {
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        // constrain to pitch
        if (constrainPitch) {
            pitch = pitch > 89.f ? 89.f : pitch;
            pitch = pitch < -89.f ? -89.f : pitch;
        }
        
        updateCameraVectors();
    }

    /*
    *  processes input data from handlers mouse scroll
    */
    void processMouseScroll(float yoffset)
    {
        zoom -= yoffset;
        if (zoom < 1.0f)
            zoom = 1.0f;
        if (zoom > 45.0f)
            zoom = 45.0f;
    }


    glm::vec3 getCamPos() const { return camPosition; }
    glm::vec3 getCamFront() const { return camFront; }
    glm::vec3 getCamUp() const { return camUp; }
    glm::vec3 getCamRight() const { return camRight; }
    glm::vec3 getWorldUp() const { return worldUp; }
    float getCamYaw() const { return yaw; }
    float getCamPitch() const { return pitch; }
    float getCamSens() const { return sensitivity; }
    float getCamMVspeed() const { return mvspeed; }
    float getCamZoom() const { return zoom; }

    void setCamPos(const glm::vec3& cp) { camPosition = cp; }
    void setCamFront(const glm::vec3& cf) { camFront = cf; }
    void setCamUp(const glm::vec3& cu) {  camUp = cu; }
    void setCamRight(const glm::vec3& cr) {  camRight = cr; }
    void setWorldUp(const glm::vec3& wu) {  worldUp = wu; }
    void setCamYaw(const float yaw) {  this->yaw  = yaw; }
    void setCamPitch(const float pitch) { this->pitch = pitch; }
    void setCamSens(const float sens) { this->sensitivity = sens; }
    void setCamMVspeed(const float mvspeed) { this->mvspeed = mvspeed; }
    void setCamZoom(const float zoom) { this->zoom = zoom; }

private:
    // camera position
    glm::vec3 camPosition;
    glm::vec3 camFront;
    glm::vec3 camUp;
    glm::vec3 camRight;
    glm::vec3 worldUp;
    // Euler's angles for camera moves
    float yaw;
    float pitch;
    // camera options
    float sensitivity;
    float mvspeed;
    float zoom;
protected:
    // compute the front vector from the Camera's (updated Euler Angles)
    void updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        camFront = glm::normalize(front);
        // also re-calculate the Right and Up vector
        camRight = glm::normalize(glm::cross(camFront, worldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        camUp = glm::normalize(glm::cross(camRight, camFront));
    }

};
