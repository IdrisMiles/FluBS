#ifndef RIGID_H
#define RIGID_H

// OpenGL includes
#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_gl_interop.h>

#include <memory>
#include <glm/glm.hpp>
#include "Rigid/rigidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"
#include "Mesh/mesh.h"


class Rigid
{
public:
    Rigid(std::shared_ptr<RigidProperty> _rigidProperty, Mesh _mesh);
    virtual ~Rigid();



    void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    void Draw();
    void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat4 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

    std::shared_ptr<RigidProperty> GetRigidProperty();

    void MapCudaGLResources();
    void ReleaseCudaGLResources();

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

    float3 *GetViscForcePtr();
    void ReleaseViscForcePtr();

    float3 *GetSurfTenForcePtr();
    void ReleaseSurfTenForcePtr();

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

    float *GetVolumePtr();
    void ReleaseVolumePtr();

    unsigned int GetMaxCellOcc();
    void SetMaxCellOcc(const unsigned int _maxCellOcc);


private:
    void Init();
    void InitCUDAMemory();
    void InitGL();
    void InitShader();
    void InitVAO();

    void CleanUpCUDAMemory();
    void CleanUpGL();


    Mesh m_mesh;
    glm::vec3 m_colour;

    // Simulation stuff
    std::shared_ptr<RigidProperty> m_rigidProperty;
    float3 *d_positionPtr;
    float3 *d_velocityPtr;
    float *d_massPtr;
    float *d_densityPtr;
    float* d_pressurePtr;
    float* d_volumePtr;

    float3* d_pressureForcePtr;
    float3* d_viscousForcePtr;
    float3* d_surfaceTensionForcePtr;
    float3* d_gravityForcePtr;
    float3* d_externalForcePtr;
    float3* d_totalForcePtr;

    unsigned int* d_particleHashIdPtr;
    unsigned int* d_cellOccupancyPtr;
    unsigned int* d_cellParticleIdxPtr;

    unsigned int m_maxCellOcc;

    bool m_positionMapped;
    bool m_velocityMapped;
    bool m_densityMapped;
    bool m_massMapped;
    bool m_pressureMapped;


    // Rendering stuff
    QOpenGLShaderProgram m_shaderProg;
    GLuint m_vertexAttrLoc;
    GLuint m_normalAttrLoc;
    GLuint m_posAttrLoc;
    GLuint m_velAttrLoc;
    GLuint m_denAttrLoc;

    GLuint m_projMatrixLoc;
    GLuint m_mvMatrixLoc;
    GLuint m_normalMatrixLoc;
    GLuint m_lightPosLoc;
    GLuint m_colourLoc;
    GLuint m_radLoc;
    GLuint m_camPosLoc;

    QOpenGLVertexArrayObject m_vao;
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

#endif // RIGID_H