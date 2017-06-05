#include "include/Render/fluidrenderer.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions>

FluidRenderer::FluidRenderer(int _w, int _h) :
    SphParticleRenderer()
{
    m_width = _w;
    m_height = _h;

    m_colour = glm::vec3(0.2f, 0.5f, 0.9f);
}

//--------------------------------------------------------------------------------------------------------------------

FluidRenderer::~FluidRenderer()
{
    CleanUpGL();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::SetSphParticles(std::shared_ptr<BaseSphParticle> _sphParticles)
{
    m_sphParticles = _sphParticles;
    m_posBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetPosBO());
    m_velBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetVelBO());
    m_denBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetDenBO());
    m_pressBO = std::make_shared<QOpenGLBuffer>(m_sphParticles->GetPressBO());

    Init();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::Draw()
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


    // Render Fluid
    int texId = 0;
    m_fluidShader.bind();
    m_fluidShader.setUniformValue("uDepthTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0 + texId++);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_smoothDepthFBO->texture());
    m_fluidShader.setUniformValue("uThicknessTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0+ texId++);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_thicknessFBO->texture());
    m_fluidShader.setUniformValue("uCubeMapTex", texId);
    glFuncs->glActiveTexture(GL_TEXTURE0+ texId++);
    glFuncs->glBindTexture(GL_TEXTURE_CUBE_MAP, m_cubeMapTex->textureId());
    m_quadVAO.bind();
    glFuncs->glDrawArrays(GL_TRIANGLES, 0, 6);
    m_fluidShader.release();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::SetShaderUniforms(const glm::mat4 &_projMat,
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

    m_fluidShader.bind();
    glFuncs->glUniform3f(m_fluidShader.uniformLocation("uCameraPos"), _camPos.x, _camPos.y, _camPos.z);
    m_fluidShader.release();

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

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::SetFrameSize(int _w, int _h)
{
    m_width=_w; m_height=_h;
    InitFBOs();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::SetCubeMap(std::shared_ptr<QOpenGLTexture> _cubemap)
{
    m_cubeMapTex = _cubemap;
}


//--------------------------------------------------------------------------------------------------------------------


void FluidRenderer::Init()
{
    InitGL();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::InitGL()
{
    InitShader();
    InitFluidVAO();
    InitQuadVAO();
    InitFBOs();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::InitShader()
{
    CreateDefaultParticleShader();
    CreateDepthShader();
    CreateSmoothDepthShader();
    CreateThicknessShader();
    CreateFluidShader();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::InitFluidVAO()
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


    // Set up pressure buffer object
    m_pressBO->bind();
    m_pressBO->release();

    m_vao.release();

    m_shaderProg.release();

}


void FluidRenderer::InitQuadVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_fluidShader.bind();
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
    glFuncs->glEnableVertexAttribArray(m_fluidShader.attributeLocation("vPos"));
    glFuncs->glVertexAttribPointer(m_fluidShader.attributeLocation("vPos"), 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
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
    glFuncs->glEnableVertexAttribArray(m_fluidShader.attributeLocation("vUV"));
    glFuncs->glVertexAttribPointer(m_fluidShader.attributeLocation("vUV"), 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), 0);
    m_quadUVBO.release();

    m_quadVAO.release();
    m_fluidShader.release();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::CleanUpGL()
{
    m_posBO.reset();
    m_velBO.reset();
    m_denBO.reset();
    m_pressBO.reset();

    m_posBO = nullptr;
    m_velBO = nullptr;
    m_denBO = nullptr;
    m_pressBO = nullptr;

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

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::InitFBOs()
{
    QOpenGLFramebufferObjectFormat fboFormat;
    fboFormat.setAttachment(QOpenGLFramebufferObject::Attachment::Depth);

    m_depthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_smoothDepthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_thicknessFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_smoothThicknessFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
}


//--------------------------------------------------------------------------------------------------------------------
// Create Shader Functions

void FluidRenderer::CreateDepthShader()
{
    // Create Depth Shader
    m_depthShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/depthVert.glsl");
    m_depthShader.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/depthGeo.glsl");
    m_depthShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/depthFrag.glsl");
    m_depthShader.link();

    // Get shader uniform and sttribute locations
    m_depthShader.bind();
    m_posAttrLoc = m_depthShader.attributeLocation("vPos");
    m_depthShader.release();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::CreateSmoothDepthShader()
{
    m_smoothDepthShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/smoothDepthVert.glsl");
    m_smoothDepthShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/smoothDepthFrag.glsl");
    m_smoothDepthShader.link();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::CreateThicknessShader()
{
    m_thicknessShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/thicknessVert.glsl");
    m_thicknessShader.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/thicknessGeo.glsl");
    m_thicknessShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/thicknessFrag.glsl");
    m_thicknessShader.link();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::CreateFluidShader()
{
    m_fluidShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/fluidVert.glsl");
    m_fluidShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/fluidFrag.glsl");
    m_fluidShader.link();
    m_fluidShader.bind();
    m_fluidShader.release();
}

//--------------------------------------------------------------------------------------------------------------------

void FluidRenderer::CreateDefaultParticleShader()
{
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/sphParticleVert.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/sphParticleGeo.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/sphParticleFrag.glsl");
    m_shaderProg.link();

    m_shaderProg.bind();
    m_posAttrLoc = m_shaderProg.attributeLocation("vPos");
    m_velAttrLoc = m_shaderProg.attributeLocation("vVel");
    m_denAttrLoc = m_shaderProg.attributeLocation("vDen");
    m_shaderProg.release();
}

//--------------------------------------------------------------------------------------------------------------------
