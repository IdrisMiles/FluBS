#include "SPH/isphparticles.h"
#include <QOpenGLContext>
#include <QOpenGLFunctions>

BaseSphParticle::BaseSphParticle()
{

}

BaseSphParticle::~BaseSphParticle()
{

}


void BaseSphParticle::SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;
    cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int));
    cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int));
}

void BaseSphParticle::Draw()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRONT_AND_BACK);

    m_shaderProg.bind();
    m_vao.bind();
    glFuncs->glDrawArrays(GL_POINTS, 0, m_property->numParticles);
    m_vao.release();
    m_shaderProg.release();

}

void BaseSphParticle::SetShaderUniforms(const glm::mat4 &_projMat, const glm::mat4 &_viewMat, const glm::mat4 &_modelMat, const glm::mat4 &_normalMat, const glm::vec3 &_lightPos, const glm::vec3 &_camPos)
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();
    m_shaderProg.bind();
    glFuncs->glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glFuncs->glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
    glFuncs->glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glFuncs->glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glFuncs->glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
    glFuncs->glUniform3fv(m_colourLoc, 1, &m_colour[0]);
    glFuncs->glUniform1f(m_radLoc, m_property->particleRadius);

    m_shaderProg.release();

}


SphParticleProperty* BaseSphParticle::GetProperty()
{
    return m_property.get();
}


//---------------------------------------------------------------------------------------------------------------


void BaseSphParticle::Init()
{
    cudaSetDevice(0);

    InitGL();
    InitCUDAMemory();

}

void BaseSphParticle::InitCUDAMemory()
{

    // particle properties
    cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_massBO_CUDA, m_massBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);

    // particle forces
    cudaMallocManaged(&d_pressureForcePtr, m_property->numParticles * sizeof(float3));
    cudaMallocManaged(&d_gravityForcePtr, m_property->numParticles * sizeof(float3));
    cudaMallocManaged(&d_externalForcePtr, m_property->numParticles * sizeof(float3));
    cudaMallocManaged(&d_totalForcePtr, m_property->numParticles * sizeof(float3));

    // particle hash
    cudaMallocManaged(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int));
}

void BaseSphParticle::InitGL()
{
    InitShader();
    InitVAO();
}

void BaseSphParticle::InitShader()
{
    // Create shaders
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/sphereSpriteVert.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/sphereSpriteGeo.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/sphereSpriteFrag.glsl");
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

void BaseSphParticle::InitVAO()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_shaderProg.bind();

    // Set up the VAO
    m_vao.create();
    m_vao.bind();


    // Setup our pos buffer object.
    m_posBO.create();
    m_posBO.bind();
    m_posBO.allocate(m_property->numParticles * sizeof(float3));
    glFuncs->glEnableVertexAttribArray(m_posAttrLoc);
    glFuncs->glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_posBO.release();


    // Set up velocity buffer object
    m_velBO.create();
    m_velBO.bind();
    m_velBO.allocate(m_property->numParticles * sizeof(float3));
    glFuncs->glEnableVertexAttribArray(m_velAttrLoc);
    glFuncs->glVertexAttribPointer(m_velAttrLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    m_velBO.release();


    // Set up density buffer object
    m_denBO.create();
    m_denBO.bind();
    m_denBO.allocate(m_property->numParticles * sizeof(float));
    glFuncs->glEnableVertexAttribArray(m_denAttrLoc);
    glFuncs->glVertexAttribPointer(m_denAttrLoc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);
    m_denBO.release();


    // Set up mass buffer object
    m_massBO.create();
    m_massBO.bind();
    m_massBO.allocate(m_property->numParticles * sizeof(float));
    m_massBO.release();


    // Set up pressure buffer object
    m_pressBO.create();
    m_pressBO.bind();
    m_pressBO.allocate(m_property->numParticles * sizeof(float));
    m_pressBO.release();


    m_vao.release();

    m_shaderProg.release();
}

void BaseSphParticle::CleanUpCUDAMemory()
{
    cudaFree(d_pressurePtr);
    cudaFree(d_pressureForcePtr);
    cudaFree(d_gravityForcePtr);
    cudaFree(d_externalForcePtr);
    cudaFree(d_totalForcePtr);
    cudaFree(d_particleHashIdPtr);
    cudaFree(d_cellOccupancyPtr);
    cudaFree(d_cellParticleIdxPtr);
}

void BaseSphParticle::CleanUpGL()
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
//---------------------------------------------------------------------------------------------------------------


void BaseSphParticle::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetMassPtr();
    GetPressurePtr();
}

void BaseSphParticle::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
    ReleaseMassPtr();
    ReleasePressurePtr();
}


float3 * BaseSphParticle::GetPositionPtr()
{
    if(!m_positionMapped)
    {
        size_t numBytes;
        cudaGraphicsMapResources(1, &m_posBO_CUDA, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_positionPtr, &numBytes, m_posBO_CUDA);

        m_positionMapped = true;
    }

    return d_positionPtr;
}

void BaseSphParticle::ReleasePositionPtr()
{
    if(m_positionMapped)
    {
        cudaGraphicsUnmapResources(1, &m_posBO_CUDA, 0);
        m_positionMapped = false;
    }

}

float3 *BaseSphParticle::GetVelocityPtr()
{
    if(!m_velocityMapped)
    {
        size_t numBytesVel;
        cudaGraphicsMapResources(1, &m_velBO_CUDA, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_velocityPtr, &numBytesVel, m_velBO_CUDA);

        m_velocityMapped = true;
    }

    return d_velocityPtr;
}

void BaseSphParticle::ReleaseVelocityPtr()
{
    if(m_velocityMapped)
    {
        cudaGraphicsUnmapResources(1, &m_velBO_CUDA, 0);

        m_velocityMapped = false;
    }
}

float *BaseSphParticle::GetDensityPtr()
{
    if(!m_densityMapped)
    {
        size_t numBytesDen;
        cudaGraphicsMapResources(1, &m_denBO_CUDA, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_densityPtr, &numBytesDen, m_denBO_CUDA);

        m_densityMapped = true;
    }

    return d_densityPtr;
}

void BaseSphParticle::ReleaseDensityPtr()
{
    if(m_densityMapped)
    {
        cudaGraphicsUnmapResources(1, &m_denBO_CUDA, 0);
        m_densityMapped = false;
    }
}

float *BaseSphParticle::GetMassPtr()
{
    if(!m_massMapped)
    {
        size_t numBytesMass;
        cudaGraphicsMapResources(1, &m_massBO_CUDA, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_massPtr, &numBytesMass, m_massBO_CUDA);

        m_massMapped = true;
    }

    return d_massPtr;
}

void BaseSphParticle::ReleaseMassPtr()
{
    if(m_massMapped)
    {
        cudaGraphicsUnmapResources(1, &m_massBO_CUDA, 0);
        m_massMapped = false;
    }
}


float *BaseSphParticle::GetPressurePtr()
{
    if(!m_pressureMapped)
    {
        size_t numBytesPress;
        cudaGraphicsMapResources(1, &m_pressBO_CUDA, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_pressurePtr, &numBytesPress, m_pressBO_CUDA);

        m_pressureMapped = true;
    }

    return d_pressurePtr;
}

void BaseSphParticle::ReleasePressurePtr()
{
    if(m_pressureMapped)
    {
        cudaGraphicsUnmapResources(1, &m_pressBO_CUDA, 0);
        m_pressureMapped = false;
    }
}

float3 *BaseSphParticle::GetPressureForcePtr()
{
    return d_pressureForcePtr;
}

void BaseSphParticle::ReleasePressureForcePtr()
{

}

float3 *BaseSphParticle::GetGravityForcePtr()
{
    return d_gravityForcePtr;
}

void BaseSphParticle::ReleaseGravityForcePtr()
{

}

float3 *BaseSphParticle::GetExternalForcePtr()
{
    return d_externalForcePtr;
}

void BaseSphParticle::ReleaseExternalForcePtr()
{

}

float3 *BaseSphParticle::GetTotalForcePtr()
{
    return d_totalForcePtr;
}

void BaseSphParticle::ReleaseTotalForcePtr()
{
}

unsigned int *BaseSphParticle::GetParticleHashIdPtr()
{
    return d_particleHashIdPtr;
}

void BaseSphParticle::ReleaseParticleHashIdPtr()
{

}

unsigned int *BaseSphParticle::GetCellOccupancyPtr()
{
    return d_cellOccupancyPtr;
}

void BaseSphParticle::ReleaseCellOccupancyPtr()
{

}

unsigned int *BaseSphParticle::GetCellParticleIdxPtr()
{
    return d_cellParticleIdxPtr;
}

void BaseSphParticle::ReleaseCellParticleIdxPtr()
{

}

unsigned int BaseSphParticle::GetMaxCellOcc()
{
    return m_maxCellOcc;
}

void BaseSphParticle::SetMaxCellOcc(const unsigned int _maxCellOcc)
{
    m_maxCellOcc = _maxCellOcc;
}
