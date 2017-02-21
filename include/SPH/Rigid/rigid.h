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
#include "SPH/isphparticles.h"
#include "SPH/Rigid/rigidproperty.h"
#include "FluidSystem/fluidsolverproperty.h"
#include "Mesh/mesh.h"


class Rigid : public ISphParticles
{
public:
    Rigid(std::shared_ptr<RigidProperty> _rigidProperty, Mesh _mesh);
    virtual ~Rigid();



    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    virtual void Draw();
    void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat4 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

    virtual RigidProperty* GetProperty();

    virtual void MapCudaGLResources();
    virtual void ReleaseCudaGLResources();


    float *GetVolumePtr();
    void ReleaseVolumePtr();

    virtual unsigned int GetMaxCellOcc();
    virtual void SetMaxCellOcc(const unsigned int _maxCellOcc);


private:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitShader();
    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();


    Mesh m_mesh;

    // Simulation stuff
    std::shared_ptr<RigidProperty> m_property;
    float* d_volumePtr;

};

#endif // RIGID_H
