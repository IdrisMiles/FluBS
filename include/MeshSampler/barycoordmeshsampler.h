#ifndef _BARYCOORDMESHSAMPLER__H_
#define _BARYCOORDMESHSAMPLER__H_

#include "Mesh/mesh.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @namespace MeshSampler
namespace MeshSampler
{

    /// @namespace MeshSampler::BaryCoord
    namespace BaryCoord
    {

        /// @brief Method to sample mesh and generate sample points
        Mesh SampleMesh(const Mesh &_mesh, const float sampleRad);//const int _numSamples);

    }

}

#endif //_BARYCOORDMESHSAMPLER__H_
