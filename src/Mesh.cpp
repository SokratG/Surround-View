#include <Mesh.hpp>



Mesh::Mesh(const std::vector<Vertex>& vertices_, const std::vector<uint>& indices_ ,
           const std::vector<Texture>& textures_, const MaterialInfo& material_) :
            vertices(vertices_), indices(indices_), textures(textures_), material(material_), VAO(0), VBO(0), EBO(0)
{
     initMesh();
}


void Mesh::initMesh()
{
    glGenVertexArrays(1, &VAO);

    glGenBuffers(1, &VBO);

    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), &indices[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texcoords));

    glBindVertexArray(0);
}


void Mesh::Draw(Shader& shader)
{
    size_t diffuseNr = 1;
    size_t specularNr = 1;
    size_t normalNr = 1;
    size_t heightNr = 1;

    for (auto i = 0u; i < textures.size(); ++i){
        glActiveTexture(GL_TEXTURE0 + i);
        std::string number;
        TexType name = textures[i].type;
        if (name == tex_DIFFUSE)
          number = std::to_string(diffuseNr++);
        else if(name == tex_SPECULAR)
          number = std::to_string(specularNr++);
        else if(name == tex_NORMAL)
          number = std::to_string(normalNr++);
        else if(name == tex_HEIGHT)
          number = std::to_string(heightNr++);

        shader.setInt((textures[i].name + number), i); // material

        glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }

    // TODO add case for Default Material
    shader.setVec3("Ka", material.ambient);
    shader.setVec3("Kd", material.diffuse);
    shader.setVec3("Ks", material.specular);
    shader.setFloat("shininess", material.shininess);


    // draw mesh
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
}


void Mesh::clearBuffers()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}


std::string TexGetNameByType(const TexType ttype)
{
    switch (ttype) {
        case tex_DIFFUSE:
            return std::string("texture_diffuse");
        case tex_SPECULAR:
            return std::string("texture_specular");
        case tex_NORMAL:
            return std::string("texture_normal");
        case tex_HEIGHT:
            return std::string("texture_height");
        default:
            return std::string();
     }
}
