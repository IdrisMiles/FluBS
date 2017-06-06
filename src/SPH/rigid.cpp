#include "SPH/rigid.h"
#include <glm/gtx/euler_angles.hpp>

Rigid::Rigid(std::shared_ptr<RigidProperty> _rigidProperty, Mesh _mesh, std::string _name, std::string _type):
    BaseSphParticle(_rigidProperty, _name)
{
    m_property = _rigidProperty;
    m_mesh = _mesh;

    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_pressureMapped = false;
    m_setupSolveSpecsInit = false;

    m_type = _type;

    m_pos = glm::vec3(0.0f, 0.0f, 0.0f);
    m_rot = glm::vec3(0.0f, 0.0f, 0.0f);

    Init();

    GetPositionPtr();
    cudaMemcpy(d_positionPtr, &m_mesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseCudaGLResources();
}


Rigid::~Rigid()
{
    m_property = nullptr;

    CleanUp();


    getLastCudaError("destructor Rigid");
}

void Rigid::UpdateMesh(Mesh &_mesh, const glm::vec3 &_pos, const glm::vec3 &_rot)
{
    m_pos = _pos;
    m_rot = _rot;
    Mesh newMesh = m_mesh = _mesh;
    for( auto &&v : newMesh.verts)
    {
        glm::mat3 t = glm::mat3(glm::eulerAngleXYZ(glm::radians(_rot.x), glm::radians(_rot.y), glm::radians(_rot.z)));
        v = (t*v)+m_pos;
    }

    if(m_property->numParticles != m_mesh.verts.size())
    {
        m_property->numParticles = m_mesh.verts.size();
        UpdateCUDAMemory();
    }

    GetPositionPtr();
    cudaMemcpy(d_positionPtr, &newMesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseCudaGLResources();
}

void Rigid::UpdateMesh(const glm::vec3 &_pos, const glm::vec3 &_rot)
{
    m_pos = _pos;
    m_rot = _rot;
    Mesh newMesh = m_mesh;
    for( auto &&v : newMesh.verts)
    {
        glm::mat3 t = glm::mat3(glm::eulerAngleXYZ(glm::radians(_rot.x), glm::radians(_rot.y), glm::radians(_rot.z)));
        v = (t*v)+m_pos;
    }

    GetPositionPtr();
    cudaMemcpy(d_positionPtr, &newMesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseCudaGLResources();
}

glm::vec3 Rigid::GetPos()
{
    return m_pos;
}

glm::vec3 Rigid::GetRot()
{
    return m_rot;
}
//---------------------------------------------------------------------------------------------------------------

RigidGpuData Rigid::GetRigidGpuData()
{
    RigidGpuData particle;
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

    particle.volume = GetVolumePtr();

    return particle;
}

//------------------------------------------------------------------------

void Rigid::SetupSolveSpecs(const FluidSolverProperty &_solverProps)
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


    getLastCudaError("SetUpSolveSpecs Rigid");

    m_setupSolveSpecsInit = true;
}

//------------------------------------------------------------------------

void Rigid::Init()
{
    cudaSetDevice(0);

    InitGL();
    InitCUDAMemory();

    m_init = true;
}

void Rigid::InitCUDAMemory()
{

    // particle properties
    cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);

    // particle forces
    cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3));

    cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int));
    cudaMalloc(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int));

    cudaMalloc(&d_volumePtr, m_property->numParticles * sizeof(float));
}

void Rigid::InitGL()
{
    InitVAO();
}


void Rigid::InitVAO()
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

//------------------------------------------------------------------------

void Rigid::CleanUpCUDAMemory()
{
    cudaFree(d_volumePtr);
}

void Rigid::CleanUpGL()
{
}

//------------------------------------------------------------------------

void Rigid::UpdateCUDAMemory()
{
    // delete memory
    checkCudaErrorsMsg(cudaFree(d_pressureForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_gravityForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_externalForcePtr),"");
    checkCudaErrorsMsg(cudaFree(d_totalForcePtr),"");

    checkCudaErrorsMsg(cudaFree(d_particleIdPtr),"");
    checkCudaErrorsMsg(cudaFree(d_particleHashIdPtr),"");

    checkCudaErrorsMsg(cudaFree(d_volumePtr),"");


    // re allocate memory
    checkCudaErrorsMsg(cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3)),"");

    checkCudaErrorsMsg(cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int)),"");
    checkCudaErrorsMsg(cudaMalloc(&d_particleIdPtr, m_property->numParticles * sizeof(unsigned int)),"");

    checkCudaErrorsMsg(cudaMalloc(&d_volumePtr, m_property->numParticles * sizeof(float)),"");


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


void Rigid::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetPressurePtr();
}

void Rigid::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
    ReleasePressurePtr();
}

//------------------------------------------------------------------------

float *Rigid::GetVolumePtr()
{
    return d_volumePtr;
}

//------------------------------------------------------------------------


RigidProperty *Rigid::GetProperty()
{
    return m_property.get();
}


//---------------------------------------------------------------------------------------------------------------

void Rigid::SetProperty(RigidProperty _property)
{
    BaseSphParticle::SetProperty(_property);

    m_property->gravity = _property.gravity;
    m_property->particleMass = _property.particleMass;
    m_property->particleRadius = _property.particleRadius;
    m_property->restDensity = _property.restDensity;
    m_property->numParticles = _property.numParticles;

    m_property->m_static = _property.m_static;
    m_property->kinematic = _property.kinematic;

    UpdateCUDAMemory();
}

//--------------------------------------------------------------------------------------------------------------------

void Rigid::SetType(std::string type)
{
    m_type = type;
}

//--------------------------------------------------------------------------------------------------------------------

std::string Rigid::GetType()
{
    return m_type;
}

//--------------------------------------------------------------------------------------------------------------------

void Rigid::SetFileName(std::string file)
{
    m_fileName = file;
}

//--------------------------------------------------------------------------------------------------------------------

std::string Rigid::GetFileName()
{
    return m_fileName;
}

