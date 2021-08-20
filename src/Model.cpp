#include <Model.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#define GL_BGR  0x80E0
#define GL_BGRA 0x80E1

uint TextureFromFile(const char* path, const std::string& directory);

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

    if (scene->HasMaterials()){
        for(auto i = 0; i < scene->mNumMaterials; ++i){
            MaterialInfo mater = processMaterial(scene->mMaterials[i]);
            materials.emplace_back(mater);
        }
    }

    processNode(scene->mRootNode, scene);
}



void Model::processNode(aiNode* node, const aiScene* scene)
{
    // process all the node’s meshes (if exists)
    for (size_t i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.emplace_back(std::move(processMesh(mesh, scene)));
    }

    // then do the same for each of it's children
    for (size_t i = 0; i < node->mNumChildren; ++i) {
            processNode(node->mChildren[i], scene);
    }

}


Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene)
{
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    std::vector<Texture> textures;

    // walk through each of the mesh's vertices
    for (size_t i = 0; i < mesh->mNumVertices; ++i) {
        glm::vec3 position{mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };
        glm::vec3 normal(0.f);
        if (mesh->HasNormals()) {
                normal = glm::vec3{ mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z };
        }
        glm::vec2 texturecoord(0.f, 0.f);
        if (mesh->mTextureCoords[0]) {
                texturecoord.x = mesh->mTextureCoords[0][i].x;
                texturecoord.y = mesh->mTextureCoords[0][i].y;
        }

        vertices.emplace_back(position, normal, texturecoord);
    }
    // now walk through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
    for (size_t i = 0; i < mesh->mNumFaces; ++i) {
            aiFace face = mesh->mFaces[i];
            for (size_t idx = 0; idx < face.mNumIndices; ++idx)
                    indices.emplace_back(face.mIndices[idx]);
    }

    if (mesh->mMaterialIndex >= 0) {
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            // 1. diffuse maps
            std::vector<Texture> diffuseMap = std::move(loadMaterialTextures(material, aiTextureType_DIFFUSE, tex_DIFFUSE));
            textures.insert(textures.end(), diffuseMap.begin(), diffuseMap.end());
            // 2. specular maps
            std::vector<Texture> specularMap = std::move(loadMaterialTextures(material, aiTextureType_SPECULAR, tex_SPECULAR));
            textures.insert(textures.end(), specularMap.begin(), specularMap.end());
            // 3. normal maps
            std::vector<Texture> normalMaps = std::move(loadMaterialTextures(material, aiTextureType_HEIGHT, tex_NORMAL));
            textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());
            // 4. height maps
            std::vector<Texture> heightMaps = loadMaterialTextures(material, aiTextureType_AMBIENT, tex_HEIGHT);
            textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());
    }

    return std::move(Mesh{ vertices, indices, textures, materials.at(mesh->mMaterialIndex) });
}


MaterialInfo Model::processMaterial(aiMaterial* material)
{
    MaterialInfo mater;
    aiString mname;
    material->Get(AI_MATKEY_NAME, mname);
    if (mname.length > 0)
      mater.name = mname.C_Str();

    int shadingModel;
    material->Get(AI_MATKEY_SHADING_MODEL, shadingModel);

    if(shadingModel != aiShadingMode_Phong && shadingModel != aiShadingMode_Gouraud){
        /* Mesh shading model is not implemented in loader, set default material and light */
        mater.name = "DefaultMaterial";
    }
    else{
        aiColor3D dif(0.f, 0.f, 0.f);
        aiColor3D amb(0.f, 0.f, 0.f);
        aiColor3D spec(0.f, 0.f, 0.f);
        float shine = 0.f;

        material->Get(AI_MATKEY_COLOR_AMBIENT, amb);
        material->Get(AI_MATKEY_COLOR_DIFFUSE, dif);
        material->Get(AI_MATKEY_COLOR_SPECULAR, spec);
        material->Get(AI_MATKEY_SHININESS, shine);

        mater.ambient = glm::vec3(amb.r, amb.g, amb.b);
        mater.diffuse = glm::vec3(dif.r, dif.g, dif.b);
        mater.specular = glm::vec3(spec.r, spec.g, spec.b);
        mater.shininess = shine;

        mater.ambient *= 0.2f;
        if(mater.shininess <= 0.f)
          mater.shininess = 32.f; // default vaule

    }


    return mater;
}





std::vector<Texture> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, const TexType typeName)
{
    std::vector<Texture> texs;

    for (size_t i = 0; i < mat->GetTextureCount(type); ++i) {
          aiString str;
          mat->GetTexture(type, i, &str);
          // skip, if texture is loaded earlier
          bool skip = false;
          for (size_t ti = 0; ti < textures_loaded.size(); ++ti) {
                  if (std::strcmp(textures_loaded[ti].path.data(), str.C_Str()) == 0) {
                          texs.push_back(textures_loaded[ti]);
                          skip = true;
                          break;
                  }
          }

          if (!skip) {
                  uint id = TextureFromFile(str.C_Str(), directory);
                  std::string name = std::move(TexGetNameByType(typeName));
                  texs.emplace_back(id, typeName, name, str.C_Str());
                  textures_loaded.emplace_back(id, typeName, name, str.C_Str());
          }
    }

    return (texs);
}


void Model::clearResource(){
    for(auto& mesh : meshes)
        mesh.clearBuffers();
}


void Model::Draw(Shader& shader)
{
    if (!isInit)
      return;

    for(auto& mesh : meshes)
      mesh.Draw(shader);
}




uint TextureFromFile(const char* path, const std::string& directory)
{
    std::string filename = std::string(path);
    filename = directory + '/' + filename;

    uint textureID = 0;
   
    cv::Mat texturedata = cv::imread(path, cv::IMREAD_UNCHANGED);

    if (texturedata.data){
        int width = texturedata.cols, height = texturedata.rows, nrComponents = texturedata.channels();

        const uchar* data = texturedata.data;

        glGenTextures(1, &textureID);

        GLenum internalformat = GL_RGB;
        GLenum dataformat = GL_RGB;
        if (nrComponents == 1)
          internalformat = dataformat = GL_RED;
        else if(nrComponents == 3){
            internalformat = dataformat = GL_RGB;
	    cv::cvtColor(texturedata, texturedata, cv::COLOR_BGR2RGB);
	}
        else if(nrComponents == 4){
          internalformat = dataformat = GL_RGBA;
	  cv::cvtColor(texturedata, texturedata, cv::COLOR_BGRA2RGBA);
	}

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalformat, width, height, 0, dataformat, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    else{
        std::cerr << "Texture failed to load at path: " << path << "\n";
        return 0;
    }

    return textureID;
}

