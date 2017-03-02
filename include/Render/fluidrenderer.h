#ifndef FLUIDRENDERER_H
#define FLUIDRENDERER_H

#include "Render/sphparticlerenderer.h"
#include <SPH/fluid.h>

#include <QOpenGLFramebufferObject>
#include <QOpenGLTexture>

class FluidRenderer : public SphParticleRenderer
{
public:
    FluidRenderer(int _w = 1280, int _h = 720);
    ~FluidRenderer();

    virtual void SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles);
    virtual void Draw();
    virtual void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat3 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

    void SetFrameSize(int _w, int _h);
    void SetCubeMap(std::shared_ptr<QOpenGLTexture> _cubemap);

protected:
    virtual void Init();
    virtual void InitGL();
    virtual void InitShader();
    virtual void InitVAO();
    virtual void CleanUpGL();

    void InitFBOs();
    void CreateDepthShader();
    void CreateSmoothDepthShader();
    void CreateThicknessShader();
    void CreateFluidShader();
    void CreateDefaultParticleShader();



    int m_width;
    int m_height;
    std::shared_ptr<QOpenGLFramebufferObject> m_depthFBO;
    std::shared_ptr<QOpenGLFramebufferObject> m_smoothDepthFBO;
    std::shared_ptr<QOpenGLFramebufferObject> m_thicknessFBO;
    std::shared_ptr<QOpenGLFramebufferObject> m_smoothThicknessFBO;

    QOpenGLShaderProgram m_depthShader;
    QOpenGLShaderProgram m_smoothDepthShader;
    QOpenGLShaderProgram m_thicknessShader;
    QOpenGLShaderProgram m_fluidShader;

    QOpenGLVertexArrayObject m_quadVAO;
    QOpenGLBuffer m_quadVBO;
    QOpenGLBuffer m_quadUVBO;

    std::shared_ptr<QOpenGLTexture> m_cubeMapTex;
};

#endif // FLUIDRENDERER_H
