#include "include/Render/bioluminescentfluidrenderer.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions>

BioluminescentFluidRenderer::BioluminescentFluidRenderer(int _w, int _h) :
    FluidRenderer(_w, _h)
{

    m_colour = glm::vec3(0.2f, 0.9f, 0.4f);
}

BioluminescentFluidRenderer::~BioluminescentFluidRenderer()
{
    CleanUpGL();
}

void BioluminescentFluidRenderer::SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles,
                                                  std::shared_ptr<Algae> _algaeParticles)
{
    m_sphParticles = _sphParticles;
    m_posBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetPosBO());
    m_velBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetVelBO());
    m_denBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetDenBO());
    m_massBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetMassBO());
    m_pressBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetPressBO());

    m_algaeParticles = _algaeParticles;
    m_algaePosBO = std::make_shared<QOpenGLBuffer>(m_algaeParticles->GetPosBO());

    Init();
}

void BioluminescentFluidRenderer::Draw()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();


    // Render Depth
    m_depthShader.bind();
    m_depthFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFuncs->glEnable(GL_DEPTH_TEST);
    glFuncs->glDisable(GL_BLEND);
    m_vao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_sphParticles->GetProperty()->numParticles);
    m_vao.release();
    m_depthFBO->release();
    m_depthShader.release();


    // Smooth depth
    m_smoothDepthShader.bind();
    m_smoothDepthFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_fluidShader.setUniformValue("uDepthTex", 0);
    glFuncs->glActiveTexture(GL_TEXTURE0);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_depthFBO->texture());
    m_quadVAO.bind();
    glFuncs->glDrawArrays(GL_TRIANGLES, 0, 6);
    m_quadVAO.release();
    m_smoothDepthFBO->release();
    m_smoothDepthShader.release();


    // Render thickness
    m_thicknessShader.bind();
    m_thicknessFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFuncs->glDisable(GL_DEPTH_TEST);
    glFuncs->glEnable(GL_BLEND);
    glFuncs->glBlendFunc(GL_ONE, GL_ONE);
    m_vao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_sphParticles->GetProperty()->numParticles);
    m_vao.release();
    glFuncs->glDisable(GL_BLEND);
    glFuncs->glEnable(GL_DEPTH_TEST);
    m_thicknessFBO->release();
    m_thicknessShader.release();


    // Render Fluid
    m_fluidShader.bind();
    m_fluidShader.setUniformValue("uDepthTex", 0);
    glFuncs->glActiveTexture(GL_TEXTURE0);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_smoothDepthFBO->texture());
    m_fluidShader.setUniformValue("uThicknessTex", 1);
    glFuncs->glActiveTexture(GL_TEXTURE0+1);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_thicknessFBO->texture());
    m_fluidShader.setUniformValue("uCubeMapTex", 2);
    glFuncs->glActiveTexture(GL_TEXTURE0+2);
    glFuncs->glBindTexture(GL_TEXTURE_CUBE_MAP, m_cubeMapTex->textureId());
    m_quadVAO.bind();
    glFuncs->glDrawArrays(GL_TRIANGLES, 0, 6);
    m_fluidShader.release();
}

void BioluminescentFluidRenderer::SetShaderUniforms(const glm::mat4 &_projMat,
                                                    const glm::mat4 &_viewMat,
                                                    const glm::mat4 &_modelMat,
                                                    const glm::mat3 &_normalMat,
                                                    const glm::vec3 &_lightPos,
                                                    const glm::vec3 &_camPos)
{

}


void BioluminescentFluidRenderer::Init()
{
    InitGL();
}

void BioluminescentFluidRenderer::InitGL()
{
    InitShader();
    InitVAO();
    InitAlgaeVAO();
    InitFBOs();
}

void BioluminescentFluidRenderer::InitShader()
{
    CreateDefaultParticleShader();
    CreateDepthShader();
    CreateSmoothDepthShader();
    CreateThicknessShader();
    CreateFluidShader();
}

void BioluminescentFluidRenderer::InitAlgaeVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_shaderProg.bind();

    // Set up the VAO
    m_algaeVao.create();
    m_algaeVao.bind();


    // Setup our alge pos buffer object.
    m_algaePosBO->bind();
    glFuncs->glEnableVertexAttribArray(m_algaePosAttrLoc);
    glFuncs->glVertexAttribPointer(m_algaePosAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_algaePosBO->release();


    // Setup our algae illumination buffer object.
    m_algaePosBO->bind();
    glFuncs->glEnableVertexAttribArray(m_algaePosAttrLoc);
    glFuncs->glVertexAttribPointer(m_algaePosAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_algaePosBO->release();


    m_algaeVao.release();

    m_shaderProg.release();

}

void BioluminescentFluidRenderer::CleanUpGL()
{
    m_posBO = nullptr;
    m_velBO = nullptr;
    m_denBO = nullptr;
    m_massBO = nullptr;
    m_pressBO = nullptr;

    m_algaePosBO = nullptr;
    m_illuminationBO = nullptr;


    m_vao.destroy();

    m_quadVBO.destroy();
    m_quadUVBO.destroy();
    m_quadVAO.destroy();

    m_depthFBO = nullptr;
    m_smoothDepthFBO = nullptr;
    m_thicknessFBO = nullptr;
    m_smoothThicknessFBO = nullptr;

    m_shaderProg.destroyed();
    m_depthShader.destroyed();
    m_smoothDepthShader.destroyed();
    m_thicknessShader.destroyed();
    m_fluidShader.destroyed();
}

void BioluminescentFluidRenderer::InitFBOs()
{

}
