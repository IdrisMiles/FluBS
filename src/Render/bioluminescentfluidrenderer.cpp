#include "include/Render/bioluminescentfluidrenderer.h"

BioluminescentFluidRenderer::BioluminescentFluidRenderer(int _w, int _h) :
    FluidRenderer(_w, _h)
{

    m_colour = glm::vec3(0.2f, 0.9f, 0.4f);
}

BioluminescentFluidRenderer::~BioluminescentFluidRenderer()
{
    CleanUpGL();
}

void BioluminescentFluidRenderer::SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles)
{
    m_sphParticles = _sphParticles;
    m_posBO.reset(m_sphParticles->GetPosBO());
    m_velBO.reset(m_sphParticles->GetVelBO());
    m_denBO.reset(m_sphParticles->GetDenBO());
    m_massBO.reset(m_sphParticles->GetMassBO());
    m_pressBO.reset(m_sphParticles->GetPressBO());

    Init();
}

void BioluminescentFluidRenderer::Draw()
{

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

}

void BioluminescentFluidRenderer::InitGL()
{

}

void BioluminescentFluidRenderer::InitShader()
{

}

void BioluminescentFluidRenderer::InitVAO()
{

}

void BioluminescentFluidRenderer::CleanUpGL()
{

}

void BioluminescentFluidRenderer::InitFBOs()
{

}
