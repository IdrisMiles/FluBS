#ifndef RIGIDRENDERER_H
#define RIGIDRENDERER_H

#include "Render/sphparticlerenderer.h"

class RigidRenderer : public SphParticleRenderer
{
public:
    RigidRenderer();
    virtual ~RigidRenderer();

    virtual void Draw();
    virtual void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat3 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

protected:

};

#endif // RIGIDRENDERER_H
