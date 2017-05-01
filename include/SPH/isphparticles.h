#ifndef ISPHPARTICLES_H
#define ISPHPARTICLES_H

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
#include "FluidSystem/fluidsolverproperty.h"
#include "Mesh/mesh.h"


class BaseSphParticle
{
public:
    BaseSphParticle(std::shared_ptr<SphParticleProperty> _property = nullptr);
    virtual ~BaseSphParticle();

    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    virtual SphParticleProperty *GetProperty();
    virtual void SetProperty(std::shared_ptr<SphParticleProperty> _property);
    virtual void SetProperty(SphParticleProperty _property);

    virtual void MapCudaGLResources();
    virtual void ReleaseCudaGLResources();

    float3 *GetPositionPtr();
    void ReleasePositionPtr();

    float3 *GetVelocityPtr();
    void ReleaseVelocityPtr();

    float *GetMassPtr();
    void ReleaseMassPtr();

    float *GetDensityPtr();
    void ReleaseDensityPtr();

    float *GetPressurePtr();
    void ReleasePressurePtr();

    float3 *GetPressureForcePtr();
    void ReleasePressureForcePtr();

    float3 *GetGravityForcePtr();
    void ReleaseGravityForcePtr();

    float3 *GetExternalForcePtr();
    void ReleaseExternalForcePtr();

    float3 *GetTotalForcePtr();
    void ReleaseTotalForcePtr();

    unsigned int *GetParticleHashIdPtr();
    void ReleaseParticleHashIdPtr();

    unsigned int *GetCellOccupancyPtr();
    void ReleaseCellOccupancyPtr();

    unsigned int *GetCellParticleIdxPtr();
    void ReleaseCellParticleIdxPtr();

    unsigned int GetMaxCellOcc();
    void SetMaxCellOcc(const unsigned int _maxCellOcc);


    QOpenGLBuffer &GetPosBO();
    QOpenGLBuffer &GetVelBO();
    QOpenGLBuffer &GetDenBO();
    QOpenGLBuffer &GetMassBO();
    QOpenGLBuffer &GetPressBO();


    virtual void GetPositions(std::vector<glm::vec3> &_pos);
    virtual void GetVelocities(std::vector<glm::vec3> &_vel);
    virtual void GetParticleIds(std::vector<int> &_ids);

    virtual void SetPositions(const std::vector<glm::vec3> &_pos);
    virtual void SetVelocities(const std::vector<glm::vec3> &_vel);
    virtual void SetParticleIds(const std::vector<int> &_ids);


protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitVAO();

    virtual void CleanUp();
    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();


    bool m_init;
    bool m_setupSolveSpecsInit;

    // Simulation Data
    Mesh m_mesh;
    std::shared_ptr<SphParticleProperty> m_property;
    float3 *d_positionPtr;
    float3 *d_velocityPtr;
    float *d_massPtr;
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
    bool m_massMapped;
    bool m_pressureMapped;

    QOpenGLBuffer m_posBO;
    QOpenGLBuffer m_velBO;
    QOpenGLBuffer m_denBO;
    QOpenGLBuffer m_massBO;
    QOpenGLBuffer m_pressBO;

    cudaGraphicsResource *m_posBO_CUDA;
    cudaGraphicsResource *m_velBO_CUDA;
    cudaGraphicsResource *m_denBO_CUDA;
    cudaGraphicsResource *m_massBO_CUDA;
    cudaGraphicsResource *m_pressBO_CUDA;
};

#endif // ISPHPARTICLES_H
