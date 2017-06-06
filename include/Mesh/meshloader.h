#ifndef MESHLOADER_H
#define MESHLOADER_H

//--------------------------------------------------------------------------------------------------------------

#include "Mesh/mesh.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class MeshLoader
/// @brief This class loads in models using ASSIMP behind the scenes
class MeshLoader
{
public:
    /// @brief constructor
    MeshLoader();

    /// @brief Static method to load a moel into a vector of meshes
    static std::vector<Mesh> LoadMesh(const std::string _meshFile);
};

//--------------------------------------------------------------------------------------------------------------

#endif // MESHLOADER_H
