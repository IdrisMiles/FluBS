#ifndef FLUIDRENDERER_H
#define FLUIDRENDERER_H

#include "Render/sphparticlerenderer.h"
#include <SPH/fluid.h>

#include <QOpenGLFramebufferObject>
#include <QOpenGLTexture>


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class FluidRenderer
/// @brief Inherits from SphParticleRenderer. This class handles rendering fluid particles as fluid using screen space fluid rendering technique
class FluidRenderer : public SphParticleRenderer
{
public:

    /// @brief Constructor
    FluidRenderer(int _w = 1280, int _h = 720);

    /// @brief destructor
    virtual ~FluidRenderer();

    /// @brief Method to set sph particles to be rendered as fluid
    virtual void SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles);

    /// @brief method to draw fluid
    virtual void Draw();

    /// @brief Method to set shader uniforms
    virtual void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat3 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

    /// @brief Method to set frame size, important as there are several full screen passes to utilize screen space rendering
    void SetFrameSize(int _w, int _h);

    /// @brief Method to set cube map used for reflections and refractions.
    void SetCubeMap(std::shared_ptr<QOpenGLTexture> _cubemap);

protected:
    virtual void Init();
    virtual void InitGL();
    virtual void InitShader();
    virtual void InitFluidVAO();
    virtual void InitQuadVAO();
    virtual void CleanUpGL();

    virtual void InitFBOs();
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

//--------------------------------------------------------------------------------------------------------------

#endif // FLUIDRENDERER_H
