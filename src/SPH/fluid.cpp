#include "SPH/fluid.h"
#include <math.h>
#include <sys/time.h>
#include <glm/gtx/transform.hpp>


Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty):
    BaseSphParticle(_fluidProperty),
    m_property(_fluidProperty)
{
    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;

    Init();
}

//--------------------------------------------------------------------------------------------------------------------

Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty, Mesh _mesh):
    BaseSphParticle(_fluidProperty),
    m_property(_fluidProperty)
{
    m_mesh = _mesh;

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;

    Init();
    InitFluidAsMesh();
}

//--------------------------------------------------------------------------------------------------------------------

Fluid::~Fluid()
{
    m_property = nullptr;

    CleanUp();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::SetupSolveSpecs(const FluidSolverProperty &_solverProps)
{
    if(m_setupSolveSpecsInit)
    {
        checkCudaErrorsMsg(cudaFree(d_cellOccupancyPtr), "Free cell occ memory in setupSolverSpecs");
        checkCudaErrorsMsg(cudaFree(d_cellParticleIdxPtr), "Free cell particle Idx memory in setupSolverSpecs");

        m_setupSolveSpecsInit = false;
    }

    const uint numCells = _solverProps.gridResolution * _solverProps.gridResolution * _solverProps.gridResolution;
    checkCudaErrorsMsg(cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int)), "Allocate cell Occ memory in setupSolverSpecs");
    checkCudaErrorsMsg(cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int)), "Allcoate cell particle Idx memory in setupSolverSpecs");


    getLastCudaError("SetUpSolveSpecs");
    m_setupSolveSpecsInit = true;
}

//--------------------------------------------------------------------------------------------------------------------

FluidProperty* Fluid::GetProperty()
{
    return m_property.get();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::Init()
{
    cudaSetDevice(0);

    InitGL();
    InitCUDAMemory();

    m_init = true;
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::InitCUDAMemory()
{

    // particle properties
    checkCudaErrorsMsg(cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsNone),"");
    checkCudaErrorsMsg(cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsNone),"");
    checkCudaErrorsMsg(cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard),"");
    checkCudaErrorsMsg(cudaGraphicsGLRegisterBuffer(&m_massBO_CUDA, m_massBO.bufferId(),cudaGraphicsMapFlagsReadOnly),"");
    checkCudaErrorsMsg(cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard),"");

    // particle forces
    checkCudaErrorsMsg(cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_viscousForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_surfaceTensionForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_predictPositionPtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_predictVelocityPtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_densityErrPtr, m_property->numParticles * sizeof(float)),"");

    // particle hash
    checkCudaErrorsMsg(cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int)),"");
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::InitGL()
{
    InitVAO();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::InitVAO()
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
}


void Fluid::InitFluidAsMesh()
{
    GetPositionPtr();
    checkCudaErrorsMsg(cudaMemcpy(d_positionPtr, &m_mesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice),"");
    ReleaseCudaGLResources();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::CleanUpCUDAMemory()
{
    checkCudaErrorsMsg(cudaFree(d_viscousForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_surfaceTensionForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_gravityForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_externalForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_totalForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_particleHashIdPtr),"");
    checkCudaErrorsMsg(cudaFree(d_cellOccupancyPtr),"");
    checkCudaErrorsMsg(cudaFree(d_cellParticleIdxPtr),"");
    checkCudaErrorsMsg(cudaFree(d_predictPositionPtr),"");
    checkCudaErrorsMsg(cudaFree(d_predictVelocityPtr),"");
    checkCudaErrorsMsg(cudaFree(d_densityErrPtr),"");
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::CleanUpGL()
{
    checkCudaErrorsMsg(cudaGraphicsUnregisterResource(m_posBO_CUDA),"");
    m_posBO.destroy();

    checkCudaErrorsMsg(cudaGraphicsUnregisterResource(m_velBO_CUDA),"");
    m_velBO.destroy();

    checkCudaErrorsMsg(cudaGraphicsUnregisterResource(m_denBO_CUDA),"");
    m_denBO.destroy();

    checkCudaErrorsMsg(cudaGraphicsUnregisterResource(m_massBO_CUDA),"");
    m_massBO.destroy();

    checkCudaErrorsMsg(cudaGraphicsUnregisterResource(m_pressBO_CUDA),"");
    m_pressBO.destroy();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetMassPtr();
    GetPressurePtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
    ReleaseMassPtr();
    ReleasePressurePtr();
}

//--------------------------------------------------------------------------------------------------------------------

float3 *Fluid::GetViscForcePtr()
{
    return d_viscousForcePtr;
}

//--------------------------------------------------------------------------------------------------------------------

float3 *Fluid::GetSurfTenForcePtr()
{
    return d_surfaceTensionForcePtr;
}

//--------------------------------------------------------------------------------------------------------------------

float3 *Fluid::GetPredictPosPtr()
{
    return d_predictPositionPtr;
}

//--------------------------------------------------------------------------------------------------------------------

float3 *Fluid::GetPredictVelPtr()
{
    return d_predictVelocityPtr;
}

//--------------------------------------------------------------------------------------------------------------------

float *Fluid::GetDensityErrPtr()
{
    return d_densityErrPtr;
}


//--------------------------------------------------------------------------------------------------------------------

void Fluid::GetPositions(std::vector<glm::vec3> &_pos)
{
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }

    _pos.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_pos[0], GetPositionPtr(), m_property->numParticles * sizeof(float3), cudaMemcpyDeviceToHost));
    ReleasePositionPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::GetVelocities(std::vector<glm::vec3> &_vel)
{
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }
    _vel.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_vel[0], GetVelocityPtr(), m_property->numParticles * sizeof(float3), cudaMemcpyDeviceToHost));
    ReleaseVelocityPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::GetParticleIds(std::vector<int> &_ids)
{
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::SetPositions(const std::vector<glm::vec3> &_pos)
{
    assert(_pos.size() == m_property->numParticles);

    cudaMemcpy(GetPositionPtr(), &_pos[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleasePositionPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::SetVelocities(const std::vector<glm::vec3> &_vel)
{
    assert(_vel.size() == m_property->numParticles);

    cudaMemcpy(GetVelocityPtr(), &_vel[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseVelocityPtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::SetParticleIds(const std::vector<int> &_ids)
{
}

//--------------------------------------------------------------------------------------------------------------------
