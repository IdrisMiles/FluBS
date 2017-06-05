#include "SPH/fluid.h"
#include <math.h>
#include <sys/time.h>
#include <glm/gtx/transform.hpp>


Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty, std::string _name):
    BaseSphParticle(_fluidProperty, _name),
    m_property(_fluidProperty)
{
    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_pressureMapped = false;
    m_setupSolveSpecsInit = false;

    Init();
}

//--------------------------------------------------------------------------------------------------------------------

Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty, Mesh _mesh, std::string _name):
    BaseSphParticle(_fluidProperty, _name),
    m_property(_fluidProperty)
{
    m_mesh = _mesh;

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
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


    getLastCudaError("SetUpSolveSpecs Fluid");
    m_setupSolveSpecsInit = true;
}

//--------------------------------------------------------------------------------------------------------------------

FluidProperty* Fluid::GetProperty()
{
    return m_property.get();
}

//---------------------------------------------------------------------------------------------------------------

void Fluid::SetProperty(FluidProperty _property)
{
    m_property->gravity = _property.gravity;
    m_property->particleMass = _property.particleMass;
    m_property->particleRadius = _property.particleRadius;
    m_property->restDensity = _property.restDensity;
    m_property->numParticles = _property.numParticles;

    m_property->gasStiffness = _property.gasStiffness;
    m_property->viscosity = _property.viscosity;
    m_property->surfaceTension= _property.surfaceTension;
    m_property->surfaceThreshold = _property.surfaceThreshold;
    m_property->numParticles = _property.numParticles;

    UpdateCUDAMemory();
}
//---------------------------------------------------------------------------------------------------------------

FluidGpuData Fluid::GetFluidGpuData()
{
    FluidGpuData particle;
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

    particle.maxCellOcc = GetMaxCellOcc();

    particle.gravity = m_property->gravity;
    particle.mass = m_property->particleMass;
    particle.restDen = m_property->restDensity;
    particle.radius = m_property->particleRadius;
    particle.smoothingLength = m_property->smoothingLength;
    particle.numParticles = m_property->numParticles;

    particle.viscousForce = GetViscForcePtr();
    particle.surfaceTensionForce = GetSurfTenForcePtr();

    particle.surfaceTension = m_property->surfaceTension;
    particle.surfaceThreshold = m_property->surfaceThreshold;
    particle.gasStiffness = m_property->gasStiffness;
    particle.viscosity = m_property->viscosity;

    return particle;
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
    checkCudaErrorsMsg(cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard),"");

    // particle forces
    checkCudaErrorsMsg(cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3)),"");

    checkCudaErrorsMsg(cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int)), "");

    checkCudaErrorsMsg(cudaMalloc(&d_viscousForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_surfaceTensionForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_predictPositionPtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_predictVelocityPtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_densityErrPtr, m_property->numParticles * sizeof(float)),"");
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
    checkCudaErrorsMsg(cudaFree(d_predictPositionPtr),"");
    checkCudaErrorsMsg(cudaFree(d_predictVelocityPtr),"");
    checkCudaErrorsMsg(cudaFree(d_densityErrPtr),"");
}

void Fluid::UpdateCUDAMemory()
{
    checkCudaErrorsMsg(cudaFree(d_pressureForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_gravityForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_externalForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_totalForcePtr),"");

    checkCudaErrorsMsg(cudaFree(d_particleIdPtr),"");
    checkCudaErrorsMsg(cudaFree(d_particleHashIdPtr),"");

    checkCudaErrorsMsg(cudaFree(d_viscousForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_surfaceTensionForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_predictPositionPtr),"");
    checkCudaErrorsMsg(cudaFree(d_predictVelocityPtr),"");
    checkCudaErrorsMsg(cudaFree(d_densityErrPtr),"");



    // particle forces
    checkCudaErrorsMsg(cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3)),"");

    checkCudaErrorsMsg(cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int)),"");

    checkCudaErrorsMsg(cudaMalloc(&d_viscousForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_surfaceTensionForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_predictPositionPtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_predictVelocityPtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_densityErrPtr, m_property->numParticles * sizeof(float)),"");


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

void Fluid::CleanUpGL()
{
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetPressurePtr();
}

//--------------------------------------------------------------------------------------------------------------------

void Fluid::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
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
    if(!m_init || this->m_property == nullptr)
    {
        return;
    }
    _ids.resize(this->m_property->numParticles);
    checkCudaErrors(cudaMemcpy(&_ids[0], GetParticleIdPtr(), this->m_property->numParticles * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

//--------------------------------------------------------------------------------------------------------------------


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
    assert(_ids.size() == m_property->numParticles);
    checkCudaErrors(cudaMemcpy(GetParticleIdPtr(), &_ids[0], m_property->numParticles * sizeof(unsigned int), cudaMemcpyHostToDevice));
}
//--------------------------------------------------------------------------------------------------------------------
