#ifndef FLUID_H
#define FLUID_H

#include "Fluid/fluidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"

// OpenGL includes
#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <memory>
#include "Fluid/isphparticles.h"
#include "Mesh/mesh.h"


class Fluid : public ISphParticles
{

public:
    Fluid(std::shared_ptr<FluidProperty> _fluidProperty);
    Fluid(std::shared_ptr<FluidProperty> _rigidProperty, Mesh _mesh);
    virtual ~Fluid();

    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    virtual void Draw();
    void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat4 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

    virtual FluidProperty *GetProperty();

    void MapCudaGLResources();
    void ReleaseCudaGLResources();

    float3 *GetViscForcePtr();
    void ReleaseViscForcePtr();

    float3 *GetSurfTenForcePtr();
    void ReleaseSurfTenForcePtr();

    virtual unsigned int GetMaxCellOcc();
    virtual void SetMaxCellOcc(const unsigned int _maxCellOcc);


protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitShader();
    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();



    Mesh m_mesh;
    // Simulation stuff
    std::shared_ptr<FluidProperty> m_fluidProperty;
    float3* d_viscousForcePtr;
    float3* d_surfaceTensionForcePtr;

};

#endif // FLUID_H
