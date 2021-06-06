#pragma once
#include <vector>
#include <string>

#include <glm.hpp>

#include <Shader.hpp>



using uint = unsigned int;

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoords;
    Vertex(const glm::vec3& position_, const glm::vec3& normal_, const glm::vec2& texcoords_) :
        position(position_), normal(normal_), texcoords(texcoords_) {}
};

typedef enum{
    tex_DIFFUSE,
    tex_SPECULAR,
    tex_NORMAL,
    tex_HEIGHT,
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


struct MaterialInfo
{
    std::string name = std::string("DefaultMaterial");
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    MaterialInfo(const glm::vec3& ambient_=glm::vec3(0.0), const glm::vec3& diffuse_=glm::vec3(0.0),
                 const glm::vec3& specular_=glm::vec3(0.0), float shininess_=0.f) :
                ambient(ambient_), diffuse(diffuse_), specular(specular_), shininess(shininess_)
    {}
};




std::string TexGetNameByType(const TexType ttype);



class Mesh
{
public:
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    std::vector<Texture> textures;
    MaterialInfo material;

public:
    Mesh(const std::vector<Vertex>& vertices_, const std::vector<uint>& indices_ ,
         const std::vector<Texture>& textures_, const MaterialInfo& material_);

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) = default;

    void Draw(Shader& shader);

    void clearBuffers();

    uint getVAO() const {return VAO;}
    uint getVBO() const {return VBO;}
    uint getEBO() const {return EBO;}
private:
    GLuint VAO, VBO, EBO;
    void initMesh();

};



