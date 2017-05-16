#include "include/Render/sphparticlerenderer.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions>

SphParticleRenderer::SphParticleRenderer()
{
    m_colour = glm::vec3(0.6f, 0.6f, 0.6f);
}

SphParticleRenderer::~SphParticleRenderer()
{
    // Get GL context
    CleanUpGL();
}

void SphParticleRenderer::SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles)
{
    m_sphParticles = _sphParticles;
    m_posBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetPosBO());
    m_velBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetVelBO());
    m_denBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetDenBO());
    m_massBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetMassBO());
    m_pressBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetPressBO());

    Init();
}

void SphParticleRenderer::Draw()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();


    glFuncs->glEnable(GL_DEPTH_TEST);

    m_shaderProg.bind();
    m_vao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_sphParticles->GetProperty()->numParticles);
    m_vao.release();
    m_shaderProg.release();
}

void SphParticleRenderer::SetShaderUniforms(const glm::mat4 &_projMat,
                                           const glm::mat4 &_viewMat,
                                           const glm::mat4 &_modelMat,
                                           const glm::mat3 &_normalMat,
                                           const glm::vec3 &_lightPos,
                                           const glm::vec3 &_camPos)
{
    if(m_sphParticles == nullptr)
    {
        return;
    }

    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();
    m_shaderProg.bind();
    glFuncs->glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glFuncs->glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glFuncs->glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
    glFuncs->glUniform3fv(m_colourLoc, 1, &m_colour[0]);
    glFuncs->glUniform1f(m_radLoc, m_sphParticles->GetProperty()->particleRadius);
    m_shaderProg.release();
}


void SphParticleRenderer::SetColour(const glm::vec3 &_colour)
{
    m_colour = _colour;
}

//-------------------------------------------------------------------------------------------------------------


void SphParticleRenderer::Init()
{
    InitGL();
}


void SphParticleRenderer::InitGL()
{
    InitShader();
    InitFluidVAO();
}

void SphParticleRenderer::InitShader()
{
    // Create shaders
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/sphParticleVert.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/sphParticleGeo.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/sphParticleFrag.glsl");
    m_shaderProg.link();

    // Get shader uniform and sttribute locations
    m_shaderProg.bind();

    m_projMatrixLoc = m_shaderProg.uniformLocation("uProjMatrix");
    m_mvMatrixLoc = m_shaderProg.uniformLocation("uMVMatrix");
    m_normalMatrixLoc = m_shaderProg.uniformLocation("uNormalMatrix");
    m_lightPosLoc = m_shaderProg.uniformLocation("uLightPos");

    m_colourLoc = m_shaderProg.uniformLocation("uColour");
    m_posAttrLoc = m_shaderProg.attributeLocation("vPos");
    m_velAttrLoc = m_shaderProg.attributeLocation("vVel");
    m_denAttrLoc = m_shaderProg.attributeLocation("vDen");
    m_radLoc = m_shaderProg.uniformLocation("uRad");
    m_camPosLoc = m_shaderProg.uniformLocation("uCameraPos");

    m_shaderProg.release();

}

void SphParticleRenderer::InitFluidVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_shaderProg.bind();

    // Set up the VAO
    m_vao.create();
    m_vao.bind();


    // Setup our pos buffer object.
    m_posBO->bind();
    glFuncs->glEnableVertexAttribArray(m_posAttrLoc);
    glFuncs->glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_posBO->release();


    // Set up velocity buffer object
    m_velBO->bind();
    glFuncs->glEnableVertexAttribArray(m_velAttrLoc);
    glFuncs->glVertexAttribPointer(m_velAttrLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    m_velBO->release();


    // Set up density buffer object
    m_denBO->bind();
    glFuncs->glEnableVertexAttribArray(m_denAttrLoc);
    glFuncs->glVertexAttribPointer(m_denAttrLoc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);
    m_denBO->release();


    // Set up mass buffer object
    m_massBO->bind();
    m_massBO->release();


    // Set up pressure buffer object
    m_pressBO->bind();
    m_pressBO->release();


    m_vao.release();

    m_shaderProg.release();
}


void SphParticleRenderer::UpdateFluidVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_shaderProg.bind();

    // Set up the VAO
    m_vao.bind();


    // Setup our pos buffer object.
    m_posBO->bind();
    glFuncs->glEnableVertexAttribArray(m_posAttrLoc);
    glFuncs->glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_posBO->release();


    // Set up velocity buffer object
    m_velBO->bind();
    glFuncs->glEnableVertexAttribArray(m_velAttrLoc);
    glFuncs->glVertexAttribPointer(m_velAttrLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    m_velBO->release();


    // Set up density buffer object
    m_denBO->bind();
    glFuncs->glEnableVertexAttribArray(m_denAttrLoc);
    glFuncs->glVertexAttribPointer(m_denAttrLoc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);
    m_denBO->release();


    // Set up mass buffer object
    m_massBO->bind();
    m_massBO->release();


    // Set up pressure buffer object
    m_pressBO->bind();
    m_pressBO->release();


    m_vao.release();

    m_shaderProg.release();
}


void SphParticleRenderer::CleanUpGL()
{

    m_posBO = nullptr;

    m_velBO = nullptr;

    m_denBO = nullptr;

    m_massBO = nullptr;

    m_pressBO = nullptr;

    m_vao.destroy();

    m_shaderProg.destroyed();
}
