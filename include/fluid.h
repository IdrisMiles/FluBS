#ifndef FLUID_H
#define FLUID_H

#include "include/sphsolverGPU.h"
#include "include/fluidproperty.h"

// OpenGL includes
#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <cuda_gl_interop.h>

// Thrust includes for CUDA stuff
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include <glm/glm.hpp>


class Fluid
{

public:
    Fluid(FluidProperty* _fluidProperty);
    ~Fluid();

    void Init();
    void Simulate();
    void Draw();
    void SetShaderUniforms(const glm::mat4 &_projMat, const glm::mat4 &_viewMat, const glm::mat4 &_modelMat, const glm::mat4 &_normalMat, const glm::vec3 &_lightPos, const glm::vec3 &_camPos);

private:
    void InitGL();
    void InitVAO();
    void InitShader();
    void InitParticles();
    void AppendSphereVerts(glm::vec3 _pos = glm::vec3(0.0f,0.0f,0.0f), float _radius = 1.0f, int _stacks = 16, int _slices = 32);


    // Simulation stuff
    SPHSolverGPU* m_solver;
    FluidProperty* m_fluidProperty;
    float3 *d_positions_ptr;
    float3 *d_velocities_ptr;
    float *d_densities_ptr;


    // Rendering stuff
    QOpenGLShaderProgram m_shaderProg;
    GLuint m_vertexAttrLoc;
    GLuint m_normalAttrLoc;
    GLuint m_posAttrLoc; // instance
    GLuint m_velAttrLoc; // instance
    GLuint m_denAttrLoc; // instance
    GLuint m_projMatrixLoc;
    GLuint m_mvMatrixLoc;
    GLuint m_normalMatrixLoc;
    GLuint m_lightPosLoc;
    GLuint m_colourLoc;
    GLuint m_radLoc;
    GLuint m_camPosLoc;

    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_meshVBO;
    QOpenGLBuffer m_meshNBO;
    QOpenGLBuffer m_meshIBO;
    QOpenGLBuffer m_posBO;
    QOpenGLBuffer m_velBO;
    QOpenGLBuffer m_denBO;

    cudaGraphicsResource *m_posBO_CUDA;
    cudaGraphicsResource *m_velBO_CUDA;
    cudaGraphicsResource *m_denBO_CUDA;

    std::vector<glm::vec3> m_meshVerts;
    std::vector<glm::vec3> m_meshNorms;
    std::vector<glm::ivec3> m_meshTris;

    glm::vec3 m_colour;

};

#endif // FLUID_H
