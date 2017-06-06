#ifndef MESH_H
#define MESH_H

#include <vector>
#include <glm/glm.hpp>

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


struct Mesh{
    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> norms;
    std::vector<glm::ivec3> tris;
};


#endif // MESH_H
