#include "SPH/basesphparticle.h"
#include <assert.h>

BaseSphParticle::BaseSphParticle(std::shared_ptr<SphParticleProperty> _property, std::string _name):
    m_property(_property),
    m_init(false),
    m_setupSolveSpecsInit(false),
    m_name(_name)
{

}

BaseSphParticle::~BaseSphParticle()
{
    m_property = nullptr;

    CleanUpCUDAMemory();
    CleanUpGL();


    getLastCudaError("destructor BASE SPH Particle");
}


void BaseSphParticle::SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps)
{
    if(m_setupSolveSpecsInit)
    {
        checkCudaErrorsMsg(cudaFree(d_cellOccupancyPtr),"");
        checkCudaErrorsMsg(cudaFree(d_cellParticleIdxPtr),"");

        m_setupSolveSpecsInit = false;
    }

    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;
    checkCudaErrorsMsg(cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int)),"");

    m_setupSolveSpecsInit = true;
}

SphParticleProperty* BaseSphParticle::GetProperty()
{
    return m_property.get();
}

//---------------------------------------------------------------------------------------------------------------

void BaseSphParticle::SetProperty(SphParticleProperty _property)
{
    m_property->gravity = _property.gravity;

    if(m_property->particleMass != _property.particleMass)
    {
        m_property->particleMass = _property.particleMass;
    }

    if(m_property->particleRadius != _property.particleRadius)
    {
        m_property->particleRadius = _property.particleRadius;
    }

    if(m_property->restDensity != _property.restDensity)
    {
        m_property->restDensity = _property.restDensity;
    }

    m_property->numParticles = _property.numParticles;

    UpdateCUDAMemory();
}

//---------------------------------------------------------------------------------------------------------------

void BaseSphParticle::SetName(const std::string _name)
{
    m_name = _name;
}

//---------------------------------------------------------------------------------------------------------------

std::string BaseSphParticle::GetName() const
{
    return m_name;
}
//---------------------------------------------------------------------------------------------------------------

ParticleGpuData BaseSphParticle::GetParticleGpuData()
{
    ParticleGpuData particle;
    particle.pos = GetPositionPtr();
    particle.vel = GetVelocityPtr();
    particle.den = GetDensityPtr();
    particle.pressure = GetPressurePtr();

    particle.pressureForce = GetPressureForcePtr();
    particle.gravityForce = GetGravityForcePtr();
    particle.externalForce = GetExternalForcePtr();
    particle.totalForce = GetTotalForcePtr();

    particle.id = GetParticleIdPtr();
    particle.hash = GetParticleHashIdPtr();
    particle.cellOcc = GetCellOccupancyPtr();
    particle.cellPartIdx = GetCellParticleIdxPtr();

    particle.gravity = m_property->gravity;
    particle.mass = m_property->particleMass;
    particle.restDen = m_property->restDensity;
    particle.radius = m_property->particleRadius;
    particle.smoothingLength = m_property->smoothingLength;
    particle.numParticles = m_property->numParticles;

    particle.maxCellOcc = GetMaxCellOcc();

    return particle;
}

//---------------------------------------------------------------------------------------------------------------


void BaseSphParticle::Init()
{
    InitGL();
    InitCUDAMemory();

    m_init = true;

}

void BaseSphParticle::InitCUDAMemory()
{

    // particle properties
    cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);

    // particle forces
    cudaMallocManaged(&d_pressureForcePtr, m_property->numParticles * sizeof(float3));
    cudaMallocManaged(&d_gravityForcePtr, m_property->numParticles * sizeof(float3));
    cudaMallocManaged(&d_externalForcePtr, m_property->numParticles * sizeof(float3));
    cudaMallocManaged(&d_totalForcePtr, m_property->numParticles * sizeof(float3));

    cudaMallocManaged(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int));
    cudaMallocManaged(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int));
}

void BaseSphParticle::InitGL()
{
    InitVAO();
}


void BaseSphParticle::InitVAO()
{
    // Setup our pos buffer object.
    m_posBO.create();
    m_posBO.bind();
    m_posBO.allocate(m_property->numParticles * sizeof(float3));
    m_posBO.release();


    // Set up velocity buffer object
    m_velBO.create();
    m_velBO.bind();
    m_velBO.allocate(m_property->numParticles * sizeof(float3));
    m_velBO.release();


    // Set up density buffer object
    m_denBO.create();
    m_denBO.bind();
    m_denBO.allocate(m_property->numParticles * sizeof(float));
    m_denBO.release();


    // Set up pressure buffer object
    m_pressBO.create();
    m_pressBO.bind();
    m_pressBO.allocate(m_property->numParticles * sizeof(float));
    m_pressBO.release();

}

void BaseSphParticle::CleanUp()
{
    CleanUpCUDAMemory();
    CleanUpGL();

    m_init = false;
}

void BaseSphParticle::CleanUpCUDAMemory()
{
    cudaFree(d_pressureForcePtr);
    cudaFree(d_gravityForcePtr);
    cudaFree(d_externalForcePtr);
    cudaFree(d_totalForcePtr);

    cudaFree(d_particleIdPtr);
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

    cudaGraphicsUnregisterResource(m_pressBO_CUDA);
    m_pressBO.destroy();

}
//---------------------------------------------------------------------------------------------------------------

void BaseSphParticle::UpdateCUDAMemory()
{
    // delete memory
    checkCudaErrorsMsg(cudaFree(d_pressureForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_gravityForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_externalForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_totalForcePtr),"");

    checkCudaErrorsMsg(cudaFree(d_particleIdPtr),"");
    checkCudaErrorsMsg(cudaFree(d_particleHashIdPtr),"");


    // re-allocate memeory
    checkCudaErrorsMsg(cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3)),"");

    checkCudaErrorsMsg(cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int)), "");


    // Setup our pos buffer object.
    m_posBO.bind();
    m_posBO.allocate(m_property->numParticles * sizeof(float3));
    m_posBO.release();

    // Set up velocity buffer object
    m_velBO.bind();
    m_velBO.allocate(m_property->numParticles * sizeof(float3));
    m_velBO.release();

    // Set up density buffer object
    m_denBO.bind();
    m_denBO.allocate(m_property->numParticles * sizeof(float));
    m_denBO.release();

    // Set up pressure buffer object
    m_pressBO.bind();
    m_pressBO.allocate(m_property->numParticles * sizeof(float));
    m_pressBO.release();
}

//--------------------------------------------------------------------------------------------------------------------


void BaseSphParticle::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetPressurePtr();
}

void BaseSphParticle::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
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

float3 *BaseSphParticle::GetGravityForcePtr()
{
    return d_gravityForcePtr;
}

float3 *BaseSphParticle::GetExternalForcePtr()
{
    return d_externalForcePtr;
}

float3 *BaseSphParticle::GetTotalForcePtr()
{
    return d_totalForcePtr;
}

unsigned int *BaseSphParticle::GetParticleHashIdPtr()
{
    return d_particleHashIdPtr;
}

unsigned int *BaseSphParticle::GetCellOccupancyPtr()
{
    return d_cellOccupancyPtr;
}

unsigned int *BaseSphParticle::GetCellParticleIdxPtr()
{
    return d_cellParticleIdxPtr;
}

unsigned int *BaseSphParticle::GetParticleIdPtr()
{
    return d_particleIdPtr;
}


unsigned int BaseSphParticle::GetMaxCellOcc()
{
    return m_maxCellOcc;
}

void BaseSphParticle::SetMaxCellOcc(const unsigned int _maxCellOcc)
{
    m_maxCellOcc = _maxCellOcc;
}


QOpenGLBuffer &BaseSphParticle::GetPosBO()
{
    return m_posBO;
}

QOpenGLBuffer &BaseSphParticle::GetVelBO()
{
    return m_velBO;
}

QOpenGLBuffer &BaseSphParticle::GetDenBO()
{
    return m_denBO;
}

QOpenGLBuffer &BaseSphParticle::GetPressBO()
{
    return m_pressBO;
}


//--------------------------------------------------------------------------------------------------------------------

void BaseSphParticle::GetPositions(std::vector<glm::vec3> &_pos)
{
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }

    _pos.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_pos[0], GetPositionPtr(), this->m_property->numParticles * sizeof(float3), cudaMemcpyDeviceToHost));
    ReleasePositionPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void BaseSphParticle::GetVelocities(std::vector<glm::vec3> &_vel)
{
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }
    _vel.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_vel[0], GetVelocityPtr(), this->m_property->numParticles * sizeof(float3), cudaMemcpyDeviceToHost));
    ReleaseVelocityPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void BaseSphParticle::GetParticleIds(std::vector<int> &_ids)
{
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }
    _ids.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_ids[0], GetParticleIdPtr(), this->m_property->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

//--------------------------------------------------------------------------------------------------------------------

void BaseSphParticle::SetPositions(const std::vector<glm::vec3> &_pos)
{
    assert(_pos.size() == m_property->numParticles);

    checkCudaErrors(cudaMemcpy(GetPositionPtr(), &_pos[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice));
    ReleasePositionPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void BaseSphParticle::SetVelocities(const std::vector<glm::vec3> &_vel)
{
    assert(_vel.size() == m_property->numParticles);

    checkCudaErrors(cudaMemcpy(GetVelocityPtr(), &_vel[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice));
    ReleaseVelocityPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void BaseSphParticle::SetParticleIds(const std::vector<int> &_ids)
{
    assert(_ids.size() == m_property->numParticles);
    checkCudaErrors(cudaMemcpy(GetParticleIdPtr(), &_ids[0], m_property->numParticles * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

//--------------------------------------------------------------------------------------------------------------------
