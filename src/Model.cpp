#include <Model.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>


void Model::InitModel(const std::string& pathmodel)
{
    if (isInit)
        return;

    loadModel(pathmodel);

    isInit = true;
}


void Model::loadModel(const std::string& path)
{
    Assimp::Importer import_;
    const aiScene* scene = import_.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
        std::cerr << "Error Assimp: " << import_.GetErrorString() << "\n";
        exit(EXIT_FAILURE);
    }

    directory = path.substr(0, path.find_last_of('/'));

    // TODO
}
