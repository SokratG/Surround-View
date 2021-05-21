#pragma once
#include <Mesh.hpp>

#include <assimp/scene.h>


using uchar = unsigned char;



class Model
{
public:
    Model() : isInit(false) {}
    Model(const std::string& pathmodel) : isInit(false) {InitModel(pathmodel);}

    void InitModel(const std::string& pathmodel);
    void Draw(Shader& shader);

    void clearResource();

    bool getModelInit() const {return isInit;}
    size_t getModelTexturesSize() const {return textures_loaded.size();}
    size_t getModelMeshesSize() const {return meshes.size();}
    const Mesh& getMesh(const uint idx) {return meshes[idx];}
    const Texture& getTexture(const uint idx) {return textures_loaded[idx];}


private:
    void loadModel(const std::string& path);
    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);
    MaterialInfo processMaterial(aiMaterial* material);
    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, const TexType typeName);
private:
    std::vector<Texture> textures_loaded;
    std::vector<Mesh> meshes;
    std::vector<MaterialInfo> materials;
    std::string directory;
    bool isInit;
};
