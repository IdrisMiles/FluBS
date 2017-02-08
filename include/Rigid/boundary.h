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
#include <device_functions.h>
#include <cuda_gl_interop.h>


#include <glm/glm.hpp>
#include "Fluid/fluid.h"
#include "Mesh/mesh.h"


class Boundary : public Fluid
{
public:
    Boundary(std::shared_ptr<FluidProperty> _fluidProperty, Mesh _mesh);
    virtual ~Boundary();


    float *GetVolumePtr();
    void ReleaseVolumePtr();

private:

    Mesh m_mesh;


    float* d_volumePtr;

};

#endif // BOUNDARY_H
