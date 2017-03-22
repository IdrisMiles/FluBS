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
    m_algaeIllumBO = std::make_shared<QOpenGLBuffer>(m_algaeParticles->GetIllumBO());

    Init();
}

void BioluminescentFluidRenderer::Draw()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    //---------------------------------------------------------------------------
    // Fluid stuff
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
    m_smoothDepthShader.setUniformValue("uDepthTex", 0);
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


    //---------------------------------------------------------------------------
    // Algae stuff
    // Render Depth
    m_depthShader.bind();
    m_algaeDepthFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFuncs->glEnable(GL_DEPTH_TEST);
    glFuncs->glDisable(GL_BLEND);
    m_algaeVao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_algaeParticles->GetProperty()->numParticles);
    m_algaeVao.release();
    m_algaeDepthFBO->release();
    m_depthShader.release();


    // Smooth depth
    m_smoothDepthShader.bind();
    m_algaeSmoothDepthFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_smoothDepthShader.setUniformValue("uDepthTex", 0);
    glFuncs->glActiveTexture(GL_TEXTURE0);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_algaeDepthFBO->texture());
    m_quadVAO.bind();
    glFuncs->glDrawArrays(GL_TRIANGLES, 0, 6);
    m_quadVAO.release();
    m_algaeSmoothDepthFBO->release();
    m_smoothDepthShader.release();


    // Render thickness
    m_biolumIntensityShader.bind();
    m_algaeThicknessFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFuncs->glDisable(GL_DEPTH_TEST);
    glFuncs->glEnable(GL_BLEND);
    glFuncs->glBlendFunc(GL_ONE, GL_ONE);
    m_algaeVao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_algaeParticles->GetProperty()->numParticles);
    m_algaeVao.release();
    glFuncs->glDisable(GL_BLEND);
    glFuncs->glEnable(GL_DEPTH_TEST);
    m_algaeThicknessFBO->release();
    m_biolumIntensityShader.release();


    //---------------------------------------------------------------------------
    // Final render
    // Render Bioluminescent Fluid
    int texId = 0;
    m_bioluminescentShader.bind();

    m_bioluminescentShader.setUniformValue("uDepthTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0 + texId++);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_smoothDepthFBO->texture());

    m_bioluminescentShader.setUniformValue("uThicknessTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0+ texId++);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_thicknessFBO->texture());

    m_bioluminescentShader.setUniformValue("uAlgaeDepthTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0 + texId++);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_algaeSmoothDepthFBO->texture());

    m_bioluminescentShader.setUniformValue("uAlgaeThicknessTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0+ texId++);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_algaeThicknessFBO->texture());

    m_bioluminescentShader.setUniformValue("uCubeMapTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0+ texId++);
    glFuncs->glBindTexture(GL_TEXTURE_CUBE_MAP, m_cubeMapTex->textureId());

    m_quadVAO.bind();
    glFuncs->glDrawArrays(GL_TRIANGLES, 0, 6);
    m_quadVAO.release();
    m_bioluminescentShader.release();
}

void BioluminescentFluidRenderer::SetShaderUniforms(const glm::mat4 &_projMat,
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

    m_depthShader.bind();
    glFuncs->glUniformMatrix4fv(m_depthShader.uniformLocation("uProjMatrix"), 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_depthShader.uniformLocation("uMVMatrix"), 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniform3fv(m_depthShader.uniformLocation("uCameraPos"), 1, &_camPos[0]);
    glFuncs->glUniform1f(m_depthShader.uniformLocation("uRad"), m_sphParticles->GetProperty()->particleRadius);
    glFuncs->glUniform1f(m_depthShader.uniformLocation("uRestDen"), m_sphParticles->GetProperty()->restDensity);
    m_depthShader.release();

    m_smoothDepthShader.bind();
    m_smoothDepthShader.release();

    m_thicknessShader.bind();
    glFuncs->glUniformMatrix4fv(m_thicknessShader.uniformLocation("uProjMatrix"), 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_thicknessShader.uniformLocation("uMVMatrix"), 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniform3fv(m_thicknessShader.uniformLocation("uCameraPos"), 1, &_camPos[0]);
    glFuncs->glUniform1f(m_thicknessShader.uniformLocation("uRad"), m_sphParticles->GetProperty()->particleRadius);
    glFuncs->glUniform1f(m_thicknessShader.uniformLocation("uRestDen"), m_sphParticles->GetProperty()->restDensity);
    m_thicknessShader.release();

    m_biolumIntensityShader.bind();
    glFuncs->glUniformMatrix4fv(m_biolumIntensityShader.uniformLocation("uProjMatrix"), 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_biolumIntensityShader.uniformLocation("uMVMatrix"), 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniform3fv(m_biolumIntensityShader.uniformLocation("uCameraPos"), 1, &_camPos[0]);
    glFuncs->glUniform1f(m_biolumIntensityShader.uniformLocation("uRad"), m_sphParticles->GetProperty()->particleRadius);
    glFuncs->glUniform1f(m_biolumIntensityShader.uniformLocation("uRestDen"), m_sphParticles->GetProperty()->restDensity);
    m_biolumIntensityShader.release();

    m_bioluminescentShader.bind();
    glFuncs->glUniform3f(m_bioluminescentShader.uniformLocation("uCameraPos"), _camPos.x, _camPos.y, _camPos.z);
    m_bioluminescentShader.release();

    m_shaderProg.bind();
    glFuncs->glUniformMatrix4fv(m_shaderProg.uniformLocation("uProjMatrix"), 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_shaderProg.uniformLocation("uMVMatrix"), 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniform3fv(m_shaderProg.uniformLocation("uCameraPos"), 1, &_camPos[0]);
    glFuncs->glUniform1f(m_shaderProg.uniformLocation("uRad"), m_sphParticles->GetProperty()->particleRadius);
    glFuncs->glUniform3fv(m_shaderProg.uniformLocation("uLightPos"), 1, &_lightPos[0]);
    glFuncs->glUniform3fv(m_shaderProg.uniformLocation("uColour"), 1, &m_colour[0]);
    glFuncs->glUniform1f(m_shaderProg.uniformLocation("uRestDen"), m_sphParticles->GetProperty()->restDensity);
    m_shaderProg.release();
}


void BioluminescentFluidRenderer::Init()
{
    InitGL();
}

void BioluminescentFluidRenderer::InitGL()
{
    InitShader();
    InitFluidVAO();
    InitAlgaeVAO();
    InitQuadVAO();
    InitFBOs();
}

void BioluminescentFluidRenderer::InitShader()
{
    CreateDefaultParticleShader();
    CreateDepthShader();
    CreateSmoothDepthShader();
    CreateThicknessShader();
    CreateBiolumIntensityShader();
    CreateBioluminescentShader();
}

void BioluminescentFluidRenderer::InitQuadVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_bioluminescentShader.bind();
    m_quadVAO.create();
    m_quadVAO.bind();

    const GLfloat quadVerts[] = {
        -1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f
    };
    m_quadVBO.create();
    m_quadVBO.bind();
    m_quadVBO.allocate(quadVerts, 6 * 3 * sizeof(float));
    glFuncs->glEnableVertexAttribArray(m_bioluminescentShader.attributeLocation("vPos"));
    glFuncs->glVertexAttribPointer(m_bioluminescentShader.attributeLocation("vPos"), 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
    m_quadVBO.release();

    const GLfloat quadUVs[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
    };
    m_quadUVBO.create();
    m_quadUVBO.bind();
    m_quadUVBO.allocate(quadUVs,6 * 2 * sizeof(float));
    glFuncs->glEnableVertexAttribArray(m_bioluminescentShader.attributeLocation("vUV"));
    glFuncs->glVertexAttribPointer(m_bioluminescentShader.attributeLocation("vUV"), 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), 0);
    m_quadUVBO.release();

    m_quadVAO.release();
    m_bioluminescentShader.release();
}

void BioluminescentFluidRenderer::InitAlgaeVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();


    // Set up the VAO
    m_algaeVao.create();
    m_algaeVao.bind();


    // Setup our alge pos buffer object.
    m_algaePosBO->bind();
    glFuncs->glEnableVertexAttribArray(m_posAttrLoc);
    glFuncs->glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_algaePosBO->release();


    // Setup our algae illumination buffer object.
    m_algaeIllumBO->bind();
    glFuncs->glEnableVertexAttribArray(m_algaeIllumAttrLoc);
    glFuncs->glVertexAttribPointer(m_algaeIllumAttrLoc, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(GLfloat), 0);
    m_algaeIllumBO->release();

    printf("%i", m_algaeIllumAttrLoc);

    m_algaeVao.release();
}

void BioluminescentFluidRenderer::CleanUpGL()
{
    m_posBO = nullptr;
    m_velBO = nullptr;
    m_denBO = nullptr;
    m_massBO = nullptr;
    m_pressBO = nullptr;

    m_algaePosBO = nullptr;
    m_algaeIllumBO = nullptr;


    m_vao.destroy();
    m_algaeVao.destroy();

    m_quadVBO.destroy();
    m_quadUVBO.destroy();
    m_quadVAO.destroy();

    m_depthFBO = nullptr;
    m_smoothDepthFBO = nullptr;
    m_thicknessFBO = nullptr;
    m_smoothThicknessFBO = nullptr;
    m_algaeDepthFBO = nullptr;
    m_algaeSmoothDepthFBO = nullptr;
    m_algaeThicknessFBO = nullptr;

    m_shaderProg.destroyed();
    m_depthShader.destroyed();
    m_smoothDepthShader.destroyed();
    m_thicknessShader.destroyed();
    m_fluidShader.destroyed();
    m_biolumIntensityShader.destroyed();
    m_bioluminescentShader.destroyed();
}

void BioluminescentFluidRenderer::InitFBOs()
{
    QOpenGLFramebufferObjectFormat fboFormat;
    fboFormat.setAttachment(QOpenGLFramebufferObject::Attachment::Depth);

    m_depthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_smoothDepthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_thicknessFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_smoothThicknessFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));

    m_algaeDepthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_algaeSmoothDepthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_algaeThicknessFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
}

void BioluminescentFluidRenderer::CreateBiolumIntensityShader()
{
    m_biolumIntensityShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/Bioluminescent/biolumIntensity.vert");
    m_biolumIntensityShader.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/Bioluminescent/biolumIntensity.geom");
    m_biolumIntensityShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/Bioluminescent/biolumIntensity.frag");
    m_biolumIntensityShader.link();
    m_biolumIntensityShader.bind();
    m_biolumIntensityShader.release();


    m_algaeIllumAttrLoc = m_biolumIntensityShader.attributeLocation("vBio");
}

void BioluminescentFluidRenderer::CreateBioluminescentShader()
{
    m_bioluminescentShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/Bioluminescent/bioluminescent.vert");
    m_bioluminescentShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/Bioluminescent/bioluminescent.frag");
    m_bioluminescentShader.link();
    m_bioluminescentShader.bind();
    m_bioluminescentShader.release();
}
