#include <Mesh.hpp>



Mesh::Mesh(const std::vector<Vertex>& vertices_, const std::vector<uint>& indices_ , const std::vector<Texture>& textures_) :
            vertices(vertices_), indices(indices_), textures(textures_)
{

}
