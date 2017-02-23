#ifndef FLUID_H
#define FLUID_H

#include "SPH/Fluid/fluidproperty.h"
#include "SPH/isphparticles.h"
#include "FluidSystem/fluidsolverproperty.h"

// Qt OpenGL includes
#include <QOpenGLFramebufferObject>
#include <QOpenGLTexture>


class Fluid : public BaseSphParticle
{

public:
    Fluid(std::shared_ptr<FluidProperty> _fluidProperty, int _w=1280, int _h=720);
    Fluid(std::shared_ptr<FluidProperty> _rigidProperty, Mesh _mesh, int _w=1280, int _h=720);
    virtual ~Fluid();

    virtual void SetupSolveSpecs(std::shared_ptr<FluidSolverProperty> _solverProps);

    virtual void Draw();
    void SetShaderUniforms(const glm::mat4 &_projMat,
                           const glm::mat4 &_viewMat,
                           const glm::mat4 &_modelMat,
                           const glm::mat4 &_normalMat,
                           const glm::vec3 &_lightPos,
                           const glm::vec3 &_camPos);
    void SetFrameSize(int _w, int _h);

    virtual FluidProperty *GetProperty();

    void MapCudaGLResources();
    void ReleaseCudaGLResources();

    float3 *GetViscForcePtr();
    void ReleaseViscForcePtr();

    float3 *GetSurfTenForcePtr();
    void ReleaseSurfTenForcePtr();


protected:
    virtual void Init();
    virtual void InitCUDAMemory();
    virtual void InitGL();
    virtual void InitShader();
    virtual void InitVAO();

    virtual void CleanUpCUDAMemory();
    virtual void CleanUpGL();

    void InitFluidAsMesh();
    void InitFBOs();
    void CreateDepthShader();
    void CreateSmoothDepthShader();
    void CreateThicknessShader();
    void CreateFluidShader();
    void CreateDefaultParticleShader();


    // Simulation Data
    std::shared_ptr<FluidProperty> m_fluidProperty;
    float3* d_viscousForcePtr;
    float3* d_surfaceTensionForcePtr;


    //---------------------------------------------------------
    // TODO remove already rendering stuff into its own class
    //---------------------------------------------------------
    // rendering stuff
    int m_width;
    int m_height;
    std::shared_ptr<QOpenGLFramebufferObject> m_depthFBO;
    std::shared_ptr<QOpenGLFramebufferObject> m_smoothDepthFBO;
    std::shared_ptr<QOpenGLFramebufferObject> m_thicknessFBO;

    QOpenGLShaderProgram m_depthShader;
    QOpenGLShaderProgram m_smoothDepthShader;
    QOpenGLShaderProgram m_thicknessShader;
    QOpenGLShaderProgram m_fluidShader;

    QOpenGLVertexArrayObject m_quadVAO;
    QOpenGLBuffer m_quadVBO;
    QOpenGLBuffer m_quadUVBO;

};

#endif // FLUID_H
