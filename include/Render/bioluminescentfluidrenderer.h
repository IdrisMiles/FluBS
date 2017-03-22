#ifndef BIOLUMINESCENTFLUIDRENDERER_H
#define BIOLUMINESCENTFLUIDRENDERER_H

#include "Render/fluidrenderer.h"
#include <SPH/algae.h>

class BioluminescentFluidRenderer : public FluidRenderer
{
public:
    BioluminescentFluidRenderer(int _w = 1280, int _h = 720);
    virtual ~BioluminescentFluidRenderer();

    virtual void SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles,
                                 std::shared_ptr<Algae> _algaeParticles);
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
    virtual void InitQuadVAO();
    virtual void InitAlgaeVAO();
    virtual void CleanUpGL();

    virtual void InitFBOs();
    void CreateBiolumIntensityShader();
    void CreateBioluminescentShader();


    std::shared_ptr<Algae> m_algaeParticles;

    QOpenGLShaderProgram m_biolumIntensityShader;
    QOpenGLShaderProgram m_bioluminescentShader;

    std::shared_ptr<QOpenGLFramebufferObject> m_algaeDepthFBO;
    std::shared_ptr<QOpenGLFramebufferObject> m_algaeSmoothDepthFBO;
    std::shared_ptr<QOpenGLFramebufferObject> m_algaeThicknessFBO;

    QOpenGLVertexArrayObject m_algaeVao;
    std::shared_ptr<QOpenGLBuffer> m_algaePosBO;
    std::shared_ptr<QOpenGLBuffer> m_algaeIllumBO;

    GLuint m_algaePosAttrLoc;
    GLuint m_algaeIllumAttrLoc;


};

#endif // BIOLUMINESCENTFLUIDRENDERER_H
