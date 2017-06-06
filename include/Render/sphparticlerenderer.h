#ifndef SPHPARTICLERENDERER_H
#define SPHPARTICLERENDERER_H

//--------------------------------------------------------------------------------------------------------------

// OpenGL includes
#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

#include <SPH/basesphparticle.h>

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class SphParticleRenderer
/// @brief the class handles rendering of sph particles as sprites
class SphParticleRenderer
{
public:
    /// @brief constructor
    SphParticleRenderer();

    /// @brief destructor
    virtual ~SphParticleRenderer();


    /// @brief Method to set the sph particles to be rendered
    virtual void SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles);

    /// @brief Method to get the sph particles that are being rendered
    virtual std::shared_ptr<BaseSphParticle> GetSphParticles();

    /// @brief Method to draw sph particles
    virtual void Draw();

    /// @brief method to set shader uniforms
    virtual void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat3 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

    /// @brief Method to set particle colour
    void SetColour(const glm::vec3 &_colour);


protected:
    virtual void Init();
    virtual void InitGL();
    virtual void InitShader();
    virtual void InitFluidVAO();
    virtual void UpdateFluidVAO();
    virtual void CleanUpGL();

    std::shared_ptr<BaseSphParticle> m_sphParticles;

    QOpenGLVertexArrayObject m_vao;
    std::shared_ptr<QOpenGLBuffer> m_posBO;
    std::shared_ptr<QOpenGLBuffer> m_velBO;
    std::shared_ptr<QOpenGLBuffer> m_denBO;
    std::shared_ptr<QOpenGLBuffer> m_pressBO;

    QOpenGLShaderProgram m_shaderProg;
    GLuint m_posAttrLoc;
    GLuint m_velAttrLoc;
    GLuint m_denAttrLoc;
    GLuint m_radLoc;
    GLuint m_lightPosLoc;
    GLuint m_colourLoc;
    GLuint m_camPosLoc;
    GLuint m_projMatrixLoc;
    GLuint m_mvMatrixLoc;
    GLuint m_normalMatrixLoc;

    glm::vec3 m_colour;
};

#endif // SPHPARTICLERENDERER_H
