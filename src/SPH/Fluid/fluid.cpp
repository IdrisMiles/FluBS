#include "SPH/Fluid/fluid.h"
#include <math.h>
#include <sys/time.h>
#include <glm/gtx/transform.hpp>
#include <QOpenGLFramebufferObjectFormat>

Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty, int _w, int _h)
{
    m_fluidProperty = _fluidProperty;
    w=_w;
    h=_h;

    m_colour = glm::vec3(0.6f, 0.6f, 0.6f);

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;

    Init();
}

Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty, Mesh _mesh)
{

    m_mesh = _mesh;
    m_fluidProperty = _fluidProperty;

    m_colour = glm::vec3(0.6f, 0.6f, 0.6f);

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;

    Init();


    GetPositionPtr();
    cudaMemcpy(d_positionPtr, &m_mesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseCudaGLResources();
}

Fluid::~Fluid()
{
    m_fluidProperty = nullptr;
    CleanUpGL();
    CleanUpCUDAMemory();
}

//------------------------------------------------------------------------

void Fluid::SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;
    cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int));
    cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int));
}

void Fluid::Draw()
{
    m_depthFBO->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//    glEnable(GL_DEPTH_TEST);

    m_shaderProg.bind();
    m_vao.bind();
    glDrawArrays(GL_POINTS, 0, m_fluidProperty->numParticles);
    m_vao.release();
    m_shaderProg.release();
    m_depthFBO->release();

    m_pointSpriteShader.bind();
    m_quadVAO.bind();
    m_pointSpriteShader.setUniformValue("tex", 0);
    glBindTexture(GL_TEXTURE_2D, m_depthFBO->texture());
    glDrawArrays(GL_TRIANGLES, 0, 6);
    m_pointSpriteShader.release();


}

void Fluid::SetShaderUniforms(const glm::mat4 &_projMat, const glm::mat4 &_viewMat, const glm::mat4 &_modelMat, const glm::mat4 &_normalMat, const glm::vec3 &_lightPos, const glm::vec3 &_camPos)
{
    m_shaderProg.bind();
    glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
    glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
    glUniform3fv(m_colourLoc, 1, &m_colour[0]);
    glUniform1f(m_radLoc, m_fluidProperty->particleRadius);

    m_shaderProg.release();

}

void Fluid::SetFrameSize(int _w, int _h)
{
    w=_w; h=_h;
    QOpenGLFramebufferObjectFormat fboFormat;
    fboFormat.setAttachment(QOpenGLFramebufferObject::Attachment::Depth);

    m_depthFBO.reset(new QOpenGLFramebufferObject(w, h, fboFormat));
    m_smoothDepthFBO.reset(new QOpenGLFramebufferObject(w, h, fboFormat));
    m_thicknessFBO.reset(new QOpenGLFramebufferObject(w, h, fboFormat));
}

//------------------------------------------------------------------------

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
    // Create shaders
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/sphereSpriteVert.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/Fluid/sphereSpriteGeo.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/sphereSpriteFrag.glsl");
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



    m_pointSpriteShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Fluid/passThroughVert.glsl");
    m_pointSpriteShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Fluid/blitTextureFrag.glsl");
    m_pointSpriteShader.link();
    m_pointSpriteShader.bind();
    m_pointSpriteShader.release();

}

void Fluid::InitVAO()
{
    m_shaderProg.bind();

    // Set up the VAO
    m_vao.create();
    m_vao.bind();


    // Setup our pos buffer object.
    m_posBO.create();
    m_posBO.bind();
    m_posBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    glEnableVertexAttribArray(m_posAttrLoc);
    glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_posBO.release();


    // Set up velocity buffer object
    m_velBO.create();
    m_velBO.bind();
    m_velBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    glEnableVertexAttribArray(m_velAttrLoc);
    glVertexAttribPointer(m_velAttrLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    m_velBO.release();


    // Set up density buffer object
    m_denBO.create();
    m_denBO.bind();
    m_denBO.allocate(m_fluidProperty->numParticles * sizeof(float));
    glEnableVertexAttribArray(m_denAttrLoc);
    glVertexAttribPointer(m_denAttrLoc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);
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

    m_shaderProg.release();



    // Testing rendering texture to screen
    m_pointSpriteShader.bind();
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
    glEnableVertexAttribArray(m_pointSpriteShader.attributeLocation("vPos"));
    glVertexAttribPointer(m_pointSpriteShader.attributeLocation("vPos"), 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
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
    glEnableVertexAttribArray(m_pointSpriteShader.attributeLocation("vUV"));
    glVertexAttribPointer(m_pointSpriteShader.attributeLocation("vUV"), 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), 0);
    m_quadUVBO.release();

    m_quadVAO.release();
    m_pointSpriteShader.release();
}

void Fluid::InitFBOs()
{
    QOpenGLFramebufferObjectFormat fboFormat;
    fboFormat.setAttachment(QOpenGLFramebufferObject::Attachment::Depth);
    m_depthFBO.reset(new QOpenGLFramebufferObject(w, h, fboFormat));
    m_smoothDepthFBO.reset(new QOpenGLFramebufferObject(w, h, fboFormat));
    m_thicknessFBO.reset(new QOpenGLFramebufferObject(w, h, fboFormat));

}

//------------------------------------------------------------------------

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
    m_shaderProg.destroyed();
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

unsigned int Fluid::GetMaxCellOcc()
{
    return m_maxCellOcc;
}

void Fluid::SetMaxCellOcc(const unsigned int _maxCellOcc)
{
    m_maxCellOcc = _maxCellOcc;
}

FluidProperty* Fluid::GetProperty()
{
    return m_fluidProperty.get();
}

