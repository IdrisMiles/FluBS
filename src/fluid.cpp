#include "include/fluid.h"
#include <math.h>
#include <sys/time.h>
#include <glm/gtx/transform.hpp>

Fluid::Fluid(std::shared_ptr<FluidProperty> _fluidProperty)
{
    m_fluidProperty = _fluidProperty;

    m_colour = glm::vec3(0.6f, 0.6f, 0.6f);

    Init();
}

Fluid::~Fluid()
{
    m_fluidProperty = nullptr;

    cudaGraphicsUnregisterResource(m_posBO_CUDA);
    m_posBO.destroy();

    cudaGraphicsUnregisterResource(m_velBO_CUDA);
    m_velBO.destroy();

    cudaGraphicsUnregisterResource(m_denBO_CUDA);
    m_denBO.destroy();

    cudaGraphicsUnregisterResource(m_massBO_CUDA);
    m_massBO.destroy();

    m_vao.destroy();
    m_shaderProg.destroyed();
}

void Fluid::Init()
{
    cudaSetDevice(0);


    InitGL();

}

void Fluid::Draw()
{
    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_CULL_FACE);
    glEnable(GL_FRONT_AND_BACK);

    m_shaderProg.bind();
    m_vao.bind();
    glDrawArrays(GL_POINTS, 0, m_fluidProperty->numParticles);
    m_vao.release();
    m_shaderProg.release();

}


void Fluid::SetShaderUniforms(const glm::mat4 &_projMat, const glm::mat4 &_viewMat, const glm::mat4 &_modelMat, const glm::mat4 &_normalMat, const glm::vec3 &_lightPos, const glm::vec3 &_camPos)
{
    m_shaderProg.bind();
    glUniformMatrix4fv(m_projMatrixLoc, 1, false, &_projMat[0][0]);
    glUniformMatrix4fv(m_mvMatrixLoc, 1, false, &(_modelMat*_viewMat)[0][0]);
    glUniformMatrix3fv(m_normalMatrixLoc, 1, true, &_normalMat[0][0]);
    glUniform3fv(m_lightPosLoc, 1, &_lightPos[0]);
    glUniform3fv(m_camPosLoc, 1, &_camPos[0]);
    glUniform3fv(m_colourLoc, 1, &m_colour[0]);
    glUniform1f(m_radLoc, m_fluidProperty->particleRadius);

    m_shaderProg.release();

}



void Fluid::InitGL()
{
    InitShader();
    InitVAO();
}

void Fluid::InitShader()
{
    // Create shaders
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/sphereSpriteVert.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Geometry, "../shader/sphereSpriteGeo.glsl");
    m_shaderProg.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/sphereSpriteFrag.glsl");
    m_shaderProg.link();

    // Get shader uniform and sttribute locations
    m_shaderProg.bind();

    m_projMatrixLoc = m_shaderProg.uniformLocation("uProjMatrix");
    m_mvMatrixLoc = m_shaderProg.uniformLocation("uMVMatrix");
    m_normalMatrixLoc = m_shaderProg.uniformLocation("uNormalMatrix");
    m_lightPosLoc = m_shaderProg.uniformLocation("uLightPos");

    m_colourLoc = m_shaderProg.uniformLocation("uColour");
    m_posAttrLoc = m_shaderProg.attributeLocation("vPos");
    m_velAttrLoc = m_shaderProg.attributeLocation("vVel");
    m_denAttrLoc = m_shaderProg.attributeLocation("vDen");
    m_radLoc = m_shaderProg.uniformLocation("uRad");
    m_camPosLoc = m_shaderProg.uniformLocation("uCameraPos");

    m_shaderProg.release();

}

void Fluid::InitVAO()
{
    m_shaderProg.bind();

    // Set up the VAO
    m_vao.create();
    m_vao.bind();


    // Setup our pos buffer object.
    m_posBO.create();
    m_posBO.bind();
    m_posBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    glEnableVertexAttribArray(m_posAttrLoc);
    glVertexAttribPointer(m_posAttrLoc, 3, GL_FLOAT, GL_FALSE, 1 * sizeof(float3), 0);
    m_posBO.release();
    cudaGraphicsGLRegisterBuffer(&m_posBO_CUDA, m_posBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);


    // Set up velocity buffer object
    m_velBO.create();
    m_velBO.bind();
    m_velBO.allocate(m_fluidProperty->numParticles * sizeof(float3));
    glEnableVertexAttribArray(m_velAttrLoc);
    glVertexAttribPointer(m_velAttrLoc, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    m_velBO.release();
    cudaGraphicsGLRegisterBuffer(&m_velBO_CUDA, m_velBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);


    // Set up density buffer object
    m_denBO.create();
    m_denBO.bind();
    m_denBO.allocate(m_fluidProperty->numParticles * sizeof(float));
    glEnableVertexAttribArray(m_denAttrLoc);
    glVertexAttribPointer(m_denAttrLoc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);
    m_denBO.release();
    cudaGraphicsGLRegisterBuffer(&m_denBO_CUDA, m_denBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);


    // Set up mass buffer object
    m_massBO.create();
    m_massBO.bind();
    m_massBO.allocate(m_fluidProperty->numParticles * sizeof(float));
//    glEnableVertexAttribArray(m_massAttrLoc);
//    glVertexAttribPointer(m_massAttrLoc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), 0);
    m_massBO.release();
    cudaGraphicsGLRegisterBuffer(&m_massBO_CUDA, m_massBO.bufferId(),cudaGraphicsMapFlagsWriteDiscard);


    glPointSize(5);
    m_vao.release();

    m_shaderProg.release();
}

float3 * Fluid::GetPositionPtr()
{
    size_t numBytes;
    cudaGraphicsMapResources(1, &m_posBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_position_ptr, &numBytes, m_posBO_CUDA);

    return d_position_ptr;
}

void Fluid::ReleasePositionPtr()
{
    cudaGraphicsUnmapResources(1, &m_posBO_CUDA, 0);
}

float3 *Fluid::GetVelocityPtr()
{
    size_t numBytesVel;
    cudaGraphicsMapResources(1, &m_velBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_velocity_ptr, &numBytesVel, m_velBO_CUDA);

    return d_velocity_ptr;
}

void Fluid::ReleaseVelocityPtr()
{
    cudaGraphicsUnmapResources(1, &m_velBO_CUDA, 0);
}

float *Fluid::GetDensityPtr()
{
    size_t numBytesDen;
    cudaGraphicsMapResources(1, &m_denBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_density_ptr, &numBytesDen, m_denBO_CUDA);

    return d_density_ptr;
}

void Fluid::ReleaseDensityPtr()
{
    cudaGraphicsUnmapResources(1, &m_denBO_CUDA, 0);
}

float *Fluid::GetMassPtr()
{
    size_t numBytesMass;
    cudaGraphicsMapResources(1, &m_massBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_mass_ptr, &numBytesMass, m_massBO_CUDA);

    return d_density_ptr;
}

void Fluid::ReleaseMassPtr()
{
    cudaGraphicsUnmapResources(1, &m_massBO_CUDA, 0);
}


std::shared_ptr<FluidProperty> Fluid::GetFluidProperty()
{
    return m_fluidProperty;
}
