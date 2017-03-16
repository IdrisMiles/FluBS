#ifndef BIOLUMINESCENTFLUIDRENDERER_H
#define BIOLUMINESCENTFLUIDRENDERER_H

#include "Render/fluidrenderer.h"
#include <SPH/algae.h>

class BioluminescentFluidRenderer : public FluidRenderer
{
public:
    BioluminescentFluidRenderer(int _w = 1280, int _h = 720);
    virtual ~BioluminescentFluidRenderer();

    virtual void SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles);
    virtual void Draw();
    virtual void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat3 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);


protected:
    virtual void Init();
    virtual void InitGL();
    virtual void InitShader();
    virtual void InitVAO();
    virtual void CleanUpGL();

    virtual void InitFBOs();
};

#endif // BIOLUMINESCENTFLUIDRENDERER_H
