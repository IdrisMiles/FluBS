#include "Mesh/meshloader.h"
#include <iostream>


MeshLoader::MeshLoader()
{

}


std::vector<Mesh> MeshLoader::LoadMesh(const std::string _meshFile)
{
    std::vector<Mesh> mesh;

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(_meshFile,
                                             aiProcess_GenSmoothNormals |
                                             aiProcess_Triangulate |
                                             aiProcess_JoinIdenticalVertices |
                                             aiProcess_SortByPType);
    if(!scene)
    {
        std::cout<<"Error loading "<<_meshFile<<" with assimp\n";
    }
    else
    {
        if(scene->HasMeshes())
        {
            mesh.resize(1);//scene->mNumMeshes);

            unsigned int indexOffset = 0;
            for(unsigned int i=0; i<scene->mNumMeshes; i++)
            {
                unsigned int numFaces = scene->mMeshes[i]->mNumFaces;
                for(unsigned int f=0; f<numFaces; f++)
                {
                    auto face = scene->mMeshes[i]->mFaces[f];
                    mesh[0].tris.push_back(glm::ivec3(face.mIndices[0]+indexOffset, face.mIndices[1]+indexOffset, face.mIndices[2]+indexOffset));
                }
                indexOffset += 3 * numFaces;
//                indexOffset = mesh[0].tris.size() * 3;

                unsigned int numVerts = scene->mMeshes[i]->mNumVertices;
                for(unsigned int v=0; v<numVerts; v++)
                {
                    auto vert = scene->mMeshes[i]->mVertices[v];
                    auto norm = scene->mMeshes[i]->mNormals[v];
                    mesh[0].verts.push_back(glm::vec3(vert.x, vert.y, vert.z));
                    mesh[0].norms.push_back(glm::vec3(norm.x, norm.y, norm.z));
                }

            } // end for num meshes
        } // end if has mesh
    } // end if valid scene

    return mesh;
}
