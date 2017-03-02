#include "include/Render/rigidrenderer.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions>

RigidRenderer::RigidRenderer()
{

}


RigidRenderer::~RigidRenderer()
{

}


void RigidRenderer::Draw()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRONT_AND_BACK);

    m_shaderProg.bind();
    m_vao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_sphParticles->GetProperty()->numParticles);
    m_vao.release();
    m_shaderProg.release();
}


void RigidRenderer::SetShaderUniforms(const glm::mat4 &_projMat,
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
