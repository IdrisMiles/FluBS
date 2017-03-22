#include "include/SPH/algae.h"

#include <math.h>
#include <glm/gtx/transform.hpp>


Algae::Algae(std::shared_ptr<AlgaeProperty> _property):
    m_property(_property)
{
    m_positionMapped = false;
    m_velocityMapped = false;
    m_densityMapped = false;
    m_massMapped = false;
    m_pressureMapped = false;
    m_illumMapped = false;

    Init();
}

//--------------------------------------------------------------------------------------------------------------------

Algae::Algae(std::shared_ptr<AlgaeProperty> _property, Mesh _mesh):
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
    CleanUpGL();
    CleanUpCUDAMemory();
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps)
{
    const uint numCells = _solverProps->gridResolution * _solverProps->gridResolution * _solverProps->gridResolution;
    cudaMalloc(&d_cellOccupancyPtr, numCells * sizeof(unsigned int));
    cudaMalloc(&d_cellParticleIdxPtr, numCells * sizeof(unsigned int));
}

//--------------------------------------------------------------------------------------------------------------------

AlgaeProperty *Algae::GetProperty()
{
    return m_property.get();
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

    // particle hash
    cudaMalloc(&d_particleHashIdPtr, m_property->numParticles * sizeof(unsigned int));

    cudaMalloc(&d_prevPressurePtr, m_property->numParticles * sizeof(float));
    cudaMalloc(&d_energyPtr, m_property->numParticles * sizeof(float));
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
    cudaFree(d_gravityForcePtr);
    cudaFree(d_externalForcePtr);
    cudaFree(d_totalForcePtr);
    cudaFree(d_particleHashIdPtr);
    cudaFree(d_cellOccupancyPtr);
    cudaFree(d_cellParticleIdxPtr);
    cudaFree(d_prevPressurePtr);
    cudaFree(d_energyPtr);
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::CleanUpGL()
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

    cudaGraphicsUnregisterResource(m_illumBO_CUDA);
    m_illumBO.destroy();
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

float *Algae::GetEnergyPtr()
{
    return d_energyPtr;
}

//--------------------------------------------------------------------------------------------------------------------

void Algae::ReleaseEnergyPtr()
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
