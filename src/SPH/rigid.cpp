#include "SPH/rigid.h"

Rigid::Rigid(std::shared_ptr<RigidProperty> _rigidProperty, Mesh _mesh):
    BaseSphParticle()
{
    m_property = _rigidProperty;
    m_mesh = _mesh;

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


Rigid::~Rigid()
{
    m_property = nullptr;
    CleanUpCUDAMemory();
    CleanUpGL();
}

void Rigid::UpdateMesh(Mesh &_mesh)
{
    m_mesh = _mesh;


    GetPositionPtr();
    cudaMemcpy(d_positionPtr, &m_mesh.verts[0], m_property->numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    ReleaseCudaGLResources();
}

//------------------------------------------------------------------------

void Rigid::SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;
    cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int));
    cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int));
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
    cudaGraphicsGLRegisterBuffer(&m_massBO_CUDA, m_massBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&m_pressBO_CUDA, m_pressBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);

    cudaMalloc(&d_volumePtr, m_property->numParticles * sizeof(float));

    // particle forces
    cudaMalloc(&d_pressureForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_gravityForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_externalForcePtr, m_property->numParticles * sizeof(float3));
    cudaMalloc(&d_totalForcePtr, m_property->numParticles * sizeof(float3));

    // particle hash
    cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int));
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

//------------------------------------------------------------------------

void Rigid::CleanUpCUDAMemory()
{
    cudaFree(d_gravityForcePtr);
    cudaFree(d_externalForcePtr);
    cudaFree(d_totalForcePtr);
    cudaFree(d_particleHashIdPtr);
    cudaFree(d_cellOccupancyPtr);
    cudaFree(d_cellParticleIdxPtr);
    cudaFree(d_volumePtr);
}

void Rigid::CleanUpGL()
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
}

//------------------------------------------------------------------------

void Rigid::MapCudaGLResources()
{
    GetPositionPtr();
    GetVelocityPtr();
    GetDensityPtr();
    GetMassPtr();
    GetPressurePtr();
}

void Rigid::ReleaseCudaGLResources()
{
    ReleasePositionPtr();
    ReleaseVelocityPtr();
    ReleaseDensityPtr();
    ReleaseMassPtr();
    ReleasePressurePtr();
}

//------------------------------------------------------------------------

float *Rigid::GetVolumePtr()
{
    return d_volumePtr;
}

//------------------------------------------------------------------------

void Rigid::ReleaseVolumePtr()
{

}

//------------------------------------------------------------------------

RigidProperty *Rigid::GetProperty()
{
    return m_property.get();
}

//--------------------------------------------------------------------------------------------------------------------

void Rigid::GetPositions(std::vector<glm::vec3> &_pos)
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

void Rigid::GetVelocities(std::vector<glm::vec3> &_vel)
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

void Rigid::GetParticleIds(std::vector<int> &_ids)
{
}

//--------------------------------------------------------------------------------------------------------------------

