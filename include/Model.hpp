#pragma once
#include <Mesh.hpp>

class Model
{
public:
    Model() : isInit(false) {}
    Model(const std::string& pathmodel) {InitModel(pathmodel);}

    void InitModel(const std::string& pathmodel);

private:
    void loadModel(const std::string& path);


private:
    std::vector<Mesh> meshes;
    std::string directory;
    bool isInit;
};
