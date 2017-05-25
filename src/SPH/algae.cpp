#include "include/SPH/algae.h"

#include <math.h>
#include <glm/gtx/transform.hpp>


Algae::Algae(std::shared_ptr<AlgaeProperty> _property, std::string _name):
    BaseSphParticle(_property, _name),
    m_property(_property)
{
    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;
    m_illumMapped = false;

    m_setupSolveSpecsInit = false;

    Init();
}

//--------------------------------------------------------------------------------------------------------------------

Algae::Algae(std::shared_ptr<AlgaeProperty> _property, Mesh _mesh, std::string _name):
    BaseSphParticle(_property, _name),
    m_property(_property)
{
    m_mesh = _mesh;

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;
    m_illumMapped = false;

    Init();
    InitAlgaeAsMesh();
}

//--------------------------------------------------------------------------------------------------------------------

Algae::~Algae()
{
    m_property = nullptr;
    CleanUp();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::SetupSolveSpecs(const FluidSolverProperty &_solverProps)
{
    if(m_setupSolveSpecsInit)
    {
        checkCudaErrorsMsg(cudaFree(d_cellOccupancyPtr),"");
        checkCudaErrorsMsg(cudaFree(d_cellParticleIdxPtr),"");

        m_setupSolveSpecsInit = false;
    }

    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;
    checkCudaErrorsMsg(cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int)),"");


    getLastCudaError("SetUpSolveSpecs Algae");

    m_setupSolveSpecsInit = true;
}

//--------------------------------------------------------------------------------------------------------------------

AlgaeProperty *Algae::GetProperty()
{
    return m_property.get();
}

//---------------------------------------------------------------------------------------------------------------

void Algae::SetProperty(AlgaeProperty _property)
{
    m_property->gravity = _property.gravity;
    m_property->particleMass = _property.particleMass;
    m_property->particleRadius = _property.particleRadius;
    m_property->restDensity = _property.restDensity;
    m_property->numParticles = _property.numParticles;

    m_property->bioluminescenceThreshold = _property.bioluminescenceThreshold;
    m_property->reactionRate = _property.reactionRate;
    m_property->deactionRate = _property.deactionRate;

    UpdateCUDAMemory();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetMassPtr();
    GetPressurePtr();
    GetIlluminationPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
    ReleaseMassPtr();
    ReleasePressurePtr();
    ReleaseIlluminationPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::Init()
{
    cudaSetDevice(0);

    InitGL();
    InitCUDAMemory();

    m_init = true;

}

//--------------------------------------------------------------------------------------------------------------------

void Algae::InitCUDAMemory()
{

    // particle properties
    cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsNone);
    cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsNone);
    cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_massBO_CUDA, m_massBO.bufferId(),cudaGraphicsMapFlagsReadOnly);
    cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_illumBO_CUDA, m_illumBO.bufferId(),cudaGraphicsMapFlagsNone);//WriteDiscard);

    // particle forces
    cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3));

    cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int));
    cudaMalloc(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int));

    cudaMalloc(&d_prevPressurePtr, m_property->numParticles * sizeof(float));
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::InitGL()
{
    InitVAO();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::InitVAO()
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


    // Set up illum buffer object
    m_illumBO.create();
    m_illumBO.bind();
    m_illumBO.allocate(m_property->numParticles * sizeof(float));
    m_illumBO.release();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::InitAlgaeAsMesh()
{
    GetPositionPtr();
    cudaMemcpy(d_positionPtr, &m_mesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseCudaGLResources();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::CleanUpCUDAMemory()
{
    cudaFree(d_prevPressurePtr);
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::CleanUpGL()
{
    cudaGraphicsUnregisterResource(m_illumBO_CUDA);
    m_illumBO.destroy();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::UpdateCUDAMemory()
{
    // delete memory
    checkCudaErrorsMsg(cudaFree(d_pressureForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_gravityForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_externalForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_totalForcePtr),"");

    checkCudaErrorsMsg(cudaFree(d_particleHashIdPtr),"");
    checkCudaErrorsMsg(cudaFree(d_particleIdPtr),"");

    checkCudaErrorsMsg(cudaFree(d_prevPressurePtr),"");


    // re allocate memory
    checkCudaErrorsMsg(cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3)),"");

    checkCudaErrorsMsg(cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int)), "");

    checkCudaErrorsMsg(cudaMalloc(&d_prevPressurePtr, m_property->numParticles * sizeof(float)), "");


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

    // Set up mass buffer object
    m_massBO.bind();
    m_massBO.allocate(m_property->numParticles * sizeof(float));
    m_massBO.release();

    // Set up pressure buffer object
    m_pressBO.bind();
    m_pressBO.allocate(m_property->numParticles * sizeof(float));
    m_pressBO.release();

    // Set up bioluminous buffer object
    m_illumBO.bind();
    m_illumBO.allocate(m_property->numParticles * sizeof(float));
    m_illumBO.release();
}

//--------------------------------------------------------------------------------------------------------------------


float *Algae::GetPrevPressurePtr()
{
    return d_prevPressurePtr;
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::ReleasePrevPressurePtr()
{

}

//--------------------------------------------------------------------------------------------------------------------

float *Algae::GetIlluminationPtr()
{
    if(!m_illumMapped)
    {
        size_t numBytesIllum;
        cudaGraphicsMapResources(1, &m_illumBO_CUDA, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_illumPtr, &numBytesIllum, m_illumBO_CUDA);

        m_illumMapped = true;
    }

    return d_illumPtr;
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::ReleaseIlluminationPtr()
{
    if(m_illumMapped)
    {
        cudaGraphicsUnmapResources(1, &m_illumBO_CUDA, 0);
        m_illumMapped = false;
    }
}

//--------------------------------------------------------------------------------------------------------------------

QOpenGLBuffer &Algae::GetIllumBO()
{
    return m_illumBO;
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::GetBioluminescentIntensities(std::vector<float> &_bio)
{
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }

    _bio.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_bio[0], GetIlluminationPtr(), this->m_property->numParticles * sizeof(float), cudaMemcpyDeviceToHost));
    ReleaseIlluminationPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::SetBioluminescentIntensities(const std::vector<float> &_bio)
{
    assert(_bio.size() == m_property->numParticles);

    cudaMemcpy(GetIlluminationPtr(), &_bio[0], m_property->numParticles * sizeof(float), cudaMemcpyHostToDevice);
    ReleaseIlluminationPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::GetPositions(std::vector<glm::vec3> &_pos)
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

void Algae::GetVelocities(std::vector<glm::vec3> &_vel)
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

void Algae::GetParticleIds(std::vector<int> &_ids)
{
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }
    _ids.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_ids[0], GetParticleIdPtr(), this->m_property->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    ReleaseParticleIdPtr();
}

//--------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------

void Algae::SetPositions(const std::vector<glm::vec3> &_pos)
{
    assert(_pos.size() == m_property->numParticles);

    cudaMemcpy(GetPositionPtr(), &_pos[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleasePositionPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::SetVelocities(const std::vector<glm::vec3> &_vel)
{
    assert(_vel.size() == m_property->numParticles);

    cudaMemcpy(GetVelocityPtr(), &_vel[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseVelocityPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::SetParticleIds(const std::vector<int> &_ids)
{
    assert(_ids.size() == m_property->numParticles);
    checkCudaErrors(cudaMemcpy(GetParticleIdPtr(), &_ids[0], m_property->numParticles * sizeof(unsigned int), cudaMemcpyHostToDevice));
    ReleaseParticleIdPtr();
}
//--------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------
