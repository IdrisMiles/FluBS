#ifndef FLUID_H
#define FLUID_H

#include "include/sphsolverGPU.h"
#include "include/fluidproperty.h"

// OpenGL includes
#include <GL/glew.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

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

//struct Particle
//{
//    float pos[3];
//    float vel[3];
//    float density;
//};

class Fluid
{

public:
    Fluid(FluidProperty* _fluidProperty, QOpenGLShaderProgram *_shaderProg);
    ~Fluid();

    void Init();
    void Simulate();
    void Draw();

private:
    void InitGL();
    void InitParticles();
    void AppendSphereVerts(glm::vec3 _pos = glm::vec3(0.0f,0.0f,0.0f), float _radius = 1.0f, int _stacks = 16, int _slices = 32);


    // Simulation stuff
    SPHSolverGPU* m_solver;
    FluidProperty* m_fluidProperty;
    float3 *d_positions_ptr;
    float3 *d_velocities_ptr;
    float *d_densities_ptr;


    // Rendering stuff
    QOpenGLShaderProgram *m_shaderProg;
    GLuint m_vertexAttrLoc;
    GLuint m_normalAttrLoc;
    GLuint m_posAttrLoc; // instance
    GLuint m_velAttrLoc; // instance
    GLuint m_denAttrLoc; // instance
    GLuint projMatrixUniformLoc;
    GLuint viewMatrixUniformLoc;

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



};

#endif // FLUID_H
