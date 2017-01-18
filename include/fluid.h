#ifndef FLUID_H
#define FLUID_H

#include "fluidproperty.h"

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


class Fluid
{

public:
    Fluid(std::shared_ptr<FluidProperty> _fluidProperty);
    ~Fluid();

    void Draw();
    void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat4 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);

    float3 *GetPositionsPtr();
    void ReleasePositionsPtr();

    float3 *GetVelocitiesPtr();
    void ReleaseVelocitiesPtr();

    float *GetDensitiesPtr();
    void ReleaseDensitiesPtr();


private:
    void Init();
    void InitGL();
    void InitVAO();
    void InitShader();


    // Simulation stuff
    std::shared_ptr<FluidProperty> m_fluidProperty;
    float3 *d_positions_ptr;
    float3 *d_velocities_ptr;
    float *d_densities_ptr;


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

    cudaGraphicsResource *m_posBO_CUDA;
    cudaGraphicsResource *m_velBO_CUDA;
    cudaGraphicsResource *m_denBO_CUDA;


    glm::vec3 m_colour;

};

#endif // FLUID_H
