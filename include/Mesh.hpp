#pragma once
#include <vector>
#include <string>
#include <glm.hpp>

#include <shader.hpp>

#include <GLES3/gl32.h>
#include <EGL/egl.h>


using uint = unsigned int;

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoords;
    Vertex(const glm::vec3& position_, const glm::vec3& normal_, const glm::vec3& texcoords_) :
        position(position_), normal(normal_), texcoords(texcoords_) {}
};

typedef enum{
    tex_DIFFUSE,
    tex_SPECULAR,
    tex_NORMAL,
    tex_HEIHGT,
    tex_UNKNOWN
} TexType;

struct Texture
{
    GLuint id;
    TexType type;
    std::string name;
    std::string path;
    Texture(const GLuint id_, const TexType type_, const std::string& name_, const std::string& path_=std::string()) :
        id(id_), type(type_), name(name_), path(path_) {}
    Texture() : id(0), type(tex_UNKNOWN) {}
};


std::string TexGetNameByType(const TexType ttype);



class Mesh
{
public:
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    std::vector<Texture> textures;

public:
    Mesh(const std::vector<Vertex>& vertices_, const std::vector<uint>& indices_ , const std::vector<Texture>& textures_);
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Mesh(const Mesh&&) = delete;
    Mesh& operator=(Mesh&&) = delete;

private:
    GLuint VAO, VBO, EBO;

};
