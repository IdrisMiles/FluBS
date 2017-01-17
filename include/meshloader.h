#ifndef MESHLOADER_H
#define MESHLOADER_H

#include "include/mesh.h"


#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>



class MeshLoader
{
public:
    MeshLoader();

    static std::vector<Mesh> LoadMesh(const std::string _meshFile);
};

#endif // MESHLOADER_H
