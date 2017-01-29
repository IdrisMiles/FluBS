#ifndef BOUNDARY_H
#define BOUNDARY_H

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

#include <glm/glm.hpp>
#include "include/mesh.h"


class Boundary
{
public:
    Boundary();
    ~Boundary();



private:

    Mesh m_mesh;

    // Simulation stuff
    float3 *d_position_ptr;
    float3 *d_velocity_ptr;
    float *d_density_ptr;
    float *d_mass_ptr;

};

#endif // BOUNDARY_H
