#ifndef ISPHPARTICLES_H
#define ISPHPARTICLES_H

//--------------------------------------------------------------------------------------------------------------


// OpenGL includes
#include <GL/glew.h>
#include <QOpenGLBuffer>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <memory>

#include "cuda_inc/helper_cuda.h"

#include "SPH/sphparticlepropeprty.h"
#include "SPH/gpudata.h"
#include "FluidSystem/fluidsolverproperty.h"
#include "Mesh/mesh.h"


//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @data 01/06/2017
//--------------------------------------------------------------------------------------------------------------


/// @class BaseSphParticle
/// @brief This class holds all data and functionality common across any type of sph particle
class BaseSphParticle
{
public:
    /// @brief constructor
    BaseSphParticle(std::shared_ptr<SphParticleProperty> _property = nullptr, std::string _name = "SPH Particle");

    /// @brief Destructor
    virtual ~BaseSphParticle();

    //------------------------------------------------------------------------------------------------------------

    /// @brief Method to set up certain internal data dependant on the solver properties, such as the hash id, cell occupancy arrays
    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    /// @brief Method to get SphParticleProperty associated with this instance
    virtual SphParticleProperty *GetProperty();

    /// @brief Method to set an instances properties
    virtual void SetProperty(SphParticleProperty _property);

    /// @brief Method to set an instances name attribute
    void SetName(const std::string _name);

    /// @brief Method to get this instances name attribute
    std::string GetName() const;

    /// @brief Method to get GPU particle data that can be used directly in CUDA kernel,
    /// this is used within the sph library
    ParticleGpuData GetParticleGpuData();

    /// @brief Method to map CUDA resources in one call, so we can use memory also being used by OpenGL
    virtual void MapCudaGLResources();

    /// @brief Method to Release CUDA OpenGL resources all in one call.
    virtual void ReleaseCudaGLResources();

    /// @brief Method to get pointer to CUDA memory holding position data
    float3 *GetPositionPtr();

    /// @brief Method to release CUDA resource to position data and give control back to OpenGL.
    void ReleasePositionPtr();

    /// @brief Method to get pointer to CUDA memory holding velocity data
    float3 *GetVelocityPtr();

    /// @brief Method to release CUDA resource to velocity data and give control back to OpenGL.
    void ReleaseVelocityPtr();

    /// @brief Method to get pointer to CUDA memory holding Density data
    float *GetDensityPtr();

    /// @brief Method to release CUDA resource to density data and give control back to OpenGL.
    void ReleaseDensityPtr();

    /// @brief Method to get pointer to CUDA memory holding pressure data
    float *GetPressurePtr();

    /// @brief Method to release CUDA resource to pressure data and give control back to OpenGL.
    void ReleasePressurePtr();

    /// @brief Method to get pointer to CUDA memory holding pressure force data
    float3 *GetPressureForcePtr();

    /// @brief Method to get pointer to CUDA memory holding gravity force data
    float3 *GetGravityForcePtr();

    /// @brief Method to get pointer to CUDA memory holding external force data
    float3 *GetExternalForcePtr();

    /// @brief Method to get pointer to CUDA memory holding total force data
    float3 *GetTotalForcePtr();

    /// @brief Method to get pointer to CUDA memory holding hash id data
    unsigned int *GetParticleHashIdPtr();

    /// @brief Method to get pointer to CUDA memory holding cell occupancy of these particles
    unsigned int *GetCellOccupancyPtr();

    /// @brief Method to get pointer to CUDA memory holding cell particle scatter addresses
    unsigned int *GetCellParticleIdxPtr();

    /// @brief Method to get pointer to CUDA memory holding particle id data
    unsigned int *GetParticleIdPtr();

    /// @brief Method to get the max cell occupancy
    unsigned int GetMaxCellOcc();

    /// @brief Method to set the max cell occupancy
    void SetMaxCellOcc(const unsigned int _maxCellOcc);

    //------------------------------------------------------------------------------------------------------------

    /// @brief Method to get the position OpenGL Buffer
    QOpenGLBuffer &GetPosBO();

    /// @brief Method to get the velocity OpenGL Buffer
    QOpenGLBuffer &GetVelBO();

    /// @brief Method to get the Density OpenGL Buffer
    QOpenGLBuffer &GetDenBO();

    /// @brief Method to get the Preesure OpenGL Buffer
    QOpenGLBuffer &GetPressBO();

    //------------------------------------------------------------------------------------------------------------

    /// @brief Method to download position data to CPU
    virtual void GetPositions(std::vector<glm::vec3> &_pos);

    /// @brief Method to download velocity data to CPU
    virtual void GetVelocities(std::vector<glm::vec3> &_vel);

    /// @brief Method to download particle id data to CPU
    virtual void GetParticleIds(std::vector<int> &_ids);

    /// @brief Method to set the position data on the GPU from CPU data
    virtual void SetPositions(const std::vector<glm::vec3> &_pos);

    /// @brief Method to set the velocity data on the GPU from CPU data
    virtual void SetVelocities(const std::vector<glm::vec3> &_vel);

    /// @brief Method to set the particle id data on the GPU from CPU data
    virtual void SetParticleIds(const std::vector<int> &_ids);

    //------------------------------------------------------------------------------------------------------------

protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUp();
    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();

    virtual void UpdateCUDAMemory();


    bool m_init;
    bool m_setupSolveSpecsInit;

    std::string m_name;

    // Simulation Data
    Mesh m_mesh;
    std::shared_ptr<SphParticleProperty> m_property;
    float3 *d_positionPtr;
    float3 *d_velocityPtr;
    float *d_densityPtr;
    float* d_pressurePtr;

    float3* d_pressureForcePtr;
    float3* d_gravityForcePtr;
    float3* d_externalForcePtr;
    float3* d_totalForcePtr;

    unsigned int* d_particleIdPtr;
    unsigned int* d_particleHashIdPtr;
    unsigned int* d_cellOccupancyPtr;
    unsigned int* d_cellParticleIdxPtr;

    unsigned int m_maxCellOcc;

    bool m_positionMapped;
    bool m_velocityMapped;
    bool m_densityMapped;
    bool m_pressureMapped;

    QOpenGLBuffer m_posBO;
    QOpenGLBuffer m_velBO;
    QOpenGLBuffer m_denBO;
    QOpenGLBuffer m_pressBO;

    cudaGraphicsResource *m_posBO_CUDA;
    cudaGraphicsResource *m_velBO_CUDA;
    cudaGraphicsResource *m_denBO_CUDA;
    cudaGraphicsResource *m_pressBO_CUDA;
};

//--------------------------------------------------------------------------------------------------------------

#endif // ISPHPARTICLES_H
