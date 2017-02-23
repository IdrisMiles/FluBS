#include "SPH/Fluid/fluid.h"
#include <math.h>
#include <sys/time.h>
#include <glm/gtx/transform.hpp>
#include <QOpenGLFramebufferObjectFormat>
#include <QOpenGLContext>
#include <QOpenGLFunctions>


Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty, int _w, int _h)
{
    m_fluidProperty = _fluidProperty;
    m_width=_w;
    m_height=_h;

    m_colour = glm::vec3(0.6f, 0.6f, 0.6f);

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;

    Init();
}

Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty, Mesh _mesh, int _w, int _h)
{
    m_fluidProperty = _fluidProperty;
    m_mesh = _mesh;
    m_width=_w;
    m_height=_h;

    m_colour = glm::vec3(0.6f, 0.6f, 0.6f);

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;

    Init();
    InitFluidAsMesh();
}

Fluid::~Fluid()
{
    m_fluidProperty = nullptr;
    CleanUpGL();
    CleanUpCUDAMemory();
}

//------------------------------------------------------------------------

void Fluid::Draw()
{

    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();
    // Render Depth
    m_depthShader.bind();
    m_depthFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFuncs->glEnable(GL_DEPTH_TEST);
    glFuncs->glDisable(GL_BLEND);
    m_vao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_fluidProperty->numParticles);
    m_vao.release();
    m_depthFBO->release();
    m_depthShader.release();

    // Smooth depth

    // Render thickness
    m_thicknessShader.bind();
    m_thicknessFBO->bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFuncs->glDisable(GL_DEPTH_TEST);
    glFuncs->glEnable(GL_BLEND);
    glFuncs->glBlendFunc(GL_ONE, GL_ONE);
    m_vao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_fluidProperty->numParticles);
    m_vao.release();
    m_thicknessFBO->release();
    m_thicknessShader.release();




    // Test render depth buffer
    m_fluidShader.bind();
    m_quadVAO.bind();
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glFuncs->glEnable(GL_DEPTH_TEST);
    glFuncs->glDisable(GL_BLEND);
    m_fluidShader.setUniformValue("uTex", 0);
    glFuncs->glActiveTexture(GL_TEXTURE0);
    glFuncs->glBindTexture(GL_TEXTURE_2D, m_thicknessFBO->texture());
    glFuncs->glDrawArrays(GL_TRIANGLES, 0, 6);
    m_fluidShader.release();


}

void Fluid::SetShaderUniforms(const glm::mat4 &_projMat,
                              const glm::mat4 &_viewMat,
                              const glm::mat4 &_modelMat,
                              const glm::mat4 &_normalMat,
                              const glm::vec3 &_lightPos,
                              const glm::vec3 &_camPos)
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_depthShader.bind();
    glFuncs->glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glFuncs->glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glFuncs->glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
    glFuncs->glUniform1f(m_radLoc, m_fluidProperty->particleRadius);
    m_depthShader.release();

//    m_depthSmoothShader.bind();
//    glFuncs->glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
//    glFuncs->glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
//    glFuncs->glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
//    glFuncs->glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
//    glFuncs->glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
//    glFuncs->glUniform1f(m_radLoc, m_fluidProperty->particleRadius);
//    m_depthSmoothShader.release();

    m_thicknessShader.bind();
    glFuncs->glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glFuncs->glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glFuncs->glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
    glFuncs->glUniform1f(m_radLoc, m_fluidProperty->particleRadius);
    m_thicknessShader.release();

}

void Fluid::SetFrameSize(int _w, int _h)
{
    m_width=_w; m_height=_h;
    InitFBOs();
}

//------------------------------------------------------------------------

void Fluid::SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;
    cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int));
    cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int));
}

void Fluid::Init()
{
    cudaSetDevice(0);

    InitGL();
    InitCUDAMemory();

}

void Fluid::InitCUDAMemory()
{

    // particle properties
    cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_massBO_CUDA, m_massBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);

    // particle forces
    cudaMalloc(&d_pressureForcePtr, m_fluidProperty->numParticles * sizeof(float3));
    cudaMalloc(&d_viscousForcePtr, m_fluidProperty->numParticles * sizeof(float3));
    cudaMalloc(&d_surfaceTensionForcePtr, m_fluidProperty->numParticles * sizeof(float3));
    cudaMalloc(&d_gravityForcePtr, m_fluidProperty->numParticles * sizeof(float3));
    cudaMalloc(&d_externalForcePtr, m_fluidProperty->numParticles * sizeof(float3));
    cudaMalloc(&d_totalForcePtr, m_fluidProperty->numParticles * sizeof(float3));

    // particle hash
    cudaMalloc(&d_particleHashIdPtr, m_fluidProperty->numParticles * sizeof(unsigned int));
}

void Fluid::InitGL()
{
    InitShader();
    InitVAO();
    InitFBOs();
}

void Fluid::InitShader()
{
    CreateDefaultParticleShader();
    CreateDepthShader();
    CreateSmoothDepthShader();
    CreateThicknessShader();
    CreateFluidShader();
}

void Fluid::InitVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_depthShader.bind();

    // Set up the VAO
    m_vao.create();
    m_vao.bind();


    // Setup our pos buffer object.
    m_posBO.create();
    m_posBO.bind();
    m_posBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    glFuncs->glEnableVertexAttribArray(m_posAttrLoc);
    glFuncs->glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_posBO.release();


    // Set up velocity buffer object
    m_velBO.create();
    m_velBO.bind();
    m_velBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    m_velBO.release();


    // Set up density buffer object
    m_denBO.create();
    m_denBO.bind();
    m_denBO.allocate(m_fluidProperty->numParticles * sizeof(float));
    m_denBO.release();


    // Set up mass buffer object
    m_massBO.create();
    m_massBO.bind();
    m_massBO.allocate(m_fluidProperty->numParticles * sizeof(float));
    m_massBO.release();


    // Set up pressure buffer object
    m_pressBO.create();
    m_pressBO.bind();
    m_pressBO.allocate(m_fluidProperty->numParticles * sizeof(float));
    m_pressBO.release();

    m_vao.release();

    m_depthShader.release();



    //-----------------------------------------------------------------------
    // Fullscreen quad
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

void Fluid::InitFBOs()
{
    QOpenGLFramebufferObjectFormat fboFormat;
    fboFormat.setAttachment(QOpenGLFramebufferObject::Attachment::Depth);

    m_depthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_smoothDepthFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
    m_thicknessFBO.reset(new QOpenGLFramebufferObject(m_width, m_height, fboFormat));
}

void Fluid::InitFluidAsMesh()
{
    GetPositionPtr();
    cudaMemcpy(d_positionPtr, &m_mesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseCudaGLResources();
}

//------------------------------------------------------------------------
// Create Shader Functions

void Fluid::CreateDepthShader()
{
    // Create Depth Shader
    m_depthShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/depthVert.glsl");
    m_depthShader.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/depthGeo.glsl");
    m_depthShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/depthFrag.glsl");
    m_depthShader.link();

    // Get shader uniform and sttribute locations
    m_depthShader.bind();

    m_projMatrixLoc = m_depthShader.uniformLocation("uProjMatrix");
    m_mvMatrixLoc = m_depthShader.uniformLocation("uMVMatrix");
    m_normalMatrixLoc = m_depthShader.uniformLocation("uNormalMatrix");
    m_lightPosLoc = m_depthShader.uniformLocation("uLightPos");
    m_posAttrLoc = m_depthShader.attributeLocation("vPos");
    m_radLoc = m_depthShader.uniformLocation("uRad");
    m_camPosLoc = m_depthShader.uniformLocation("uCameraPos");

    m_depthShader.release();
}

void Fluid::CreateSmoothDepthShader()
{
    m_smoothDepthShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/smoothDepthVert.glsl");
    m_smoothDepthShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/smoothDepthFrag.glsl");
    m_smoothDepthShader.link();
}

void Fluid::CreateThicknessShader()
{
    m_thicknessShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/thicknessVert.glsl");
    m_thicknessShader.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/thicknessGeo.glsl");
    m_thicknessShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/thicknessFrag.glsl");
    m_thicknessShader.link();
}

void Fluid::CreateFluidShader()
{
    m_fluidShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/blitTextureVert.glsl");
    m_fluidShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/blitTextureFrag.glsl");
    m_fluidShader.link();
    m_fluidShader.bind();
    m_fluidShader.release();
}

void Fluid::CreateDefaultParticleShader()
{
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/sphParticleVert.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/sphParticleGeo.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/sphParticleFrag.glsl");
    m_shaderProg.link();
}

//------------------------------------------------------------------------
// Clean-up Functions

void Fluid::CleanUpCUDAMemory()
{
    cudaFree(d_pressureForcePtr);
    cudaFree(d_viscousForcePtr);
    cudaFree(d_surfaceTensionForcePtr);
    cudaFree(d_gravityForcePtr);
    cudaFree(d_externalForcePtr);
    cudaFree(d_totalForcePtr);
    cudaFree(d_particleHashIdPtr);
    cudaFree(d_cellOccupancyPtr);
    cudaFree(d_cellParticleIdxPtr);
}

void Fluid::CleanUpGL()
{
    cudaGraphicsUnregisterResource(m_posBO_CUDA);
    m_posBO.destroy();

    cudaGraphicsUnregisterResource(m_velBO_CUDA);
    m_velBO.destroy();

    cudaGraphicsUnregisterResource(m_denBO_CUDA);
    m_denBO.destroy();

    cudaGraphicsUnregisterResource(m_massBO_CUDA);
    m_massBO.destroy();

    cudaGraphicsUnregisterResource(m_pressBO_CUDA);
    m_pressBO.destroy();

    m_vao.destroy();

    m_quadVBO.destroy();
    m_quadUVBO.destroy();
    m_quadVAO.destroy();

    m_depthFBO = nullptr;
    m_smoothDepthFBO = nullptr;
    m_thicknessFBO = nullptr;

    m_shaderProg.destroyed();
    m_depthShader.destroyed();
    m_smoothDepthShader.destroyed();
    m_thicknessShader.destroyed();
    m_fluidShader.destroyed();

}

//------------------------------------------------------------------------

void Fluid::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetMassPtr();
    GetPressurePtr();
}

void Fluid::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
    ReleaseMassPtr();
    ReleasePressurePtr();
}

//------------------------------------------------------------------------

float3 *Fluid::GetViscForcePtr()
{
    return d_viscousForcePtr;
}

void Fluid::ReleaseViscForcePtr()
{

}

float3 *Fluid::GetSurfTenForcePtr()
{
    return d_surfaceTensionForcePtr;
}

void Fluid::ReleaseSurfTenForcePtr()
{

}

FluidProperty* Fluid::GetProperty()
{
    return m_fluidProperty.get();
}


