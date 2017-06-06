#ifndef RIGIDRENDERER_H
#define RIGIDRENDERER_H

//--------------------------------------------------------------------------------------------------------------
#include "Render/sphparticlerenderer.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class RigidRenderer
/// @brief Inherits form SphParticleRenderer.This class handles rendering rigid particles as sprites
class RigidRenderer : public SphParticleRenderer
{
public:
    /// @brief constructor
    RigidRenderer();

    /// @brief destructor
    virtual ~RigidRenderer();

    /// @brief Method to draw particles
    virtual void Draw();

    /// @brief Method to set shader uniforms
    virtual void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat3 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

protected:

};

//--------------------------------------------------------------------------------------------------------------

#endif // RIGIDRENDERER_H
