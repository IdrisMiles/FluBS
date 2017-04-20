#ifndef OPENGLSCENE_H
#define OPENGLSCENE_H


#include <GL/glew.h>

#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QTimer>

#include <Cache/cachesystem.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>



#include "FluidSystem/fluidsystem.h"

#include "Render/fluidrenderer.h"
#include "Render/rigidrenderer.h"
#include "Render/bioluminescentfluidrenderer.h"

#include "Mesh/meshloader.h"
#include "Mesh/rendermesh.h"


class OpenGLScene : public QOpenGLWidget
{

    Q_OBJECT

public:
    OpenGLScene(QWidget *parent = 0);
    ~OpenGLScene();

    QSize minimumSizeHint() const Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;


public slots:
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    void setXTranslation(int x);
    void setYTranslation(int y);
    void setZTranslation(int z);
    void cleanup();
    void UpdateSim();
    void ResetSim();
    void OnFrameChanged(int frame);

signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);
    void xTranslationChanged(int x);
    void yTranslationChanged(int y);
    void zTranslationChanged(int z);
    void FluidSystemInitialised(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty);
    void FluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty);
    void RigidInitialised(std::shared_ptr<RigidProperty> _rigidProperty);
    void AlgaeInitialised(std::shared_ptr<AlgaeProperty> _algarProperty);
    void FrameCached(int frame);
    void FrameSimmed(int frame);
    void FrameLoaded(int frame);

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;

private:

    void DrawSkybox();
    void CreateSkybox();
    QOpenGLShaderProgram m_skyboxShader;
    std::shared_ptr<QOpenGLTexture> m_skyboxTex;
    QOpenGLVertexArrayObject m_skyboxVAO;
    QOpenGLBuffer m_skyboxVBO;

    /// @brief Attribute to control the x rotation of the scene
    int m_xRot;

    /// @brief Attribute to control the y rotation of the scene
    int m_yRot;

    /// @brief Attribute to control the z rotation of the scene
    int m_zRot;

    /// @brief Attribute to control the x translation of the scene
    int m_xDis;

    /// @brief Attribute to control the y translation of the scene
    int m_yDis;

    /// @brief Attribute to control the z translation of the scene
    int m_zDis;

    glm::mat4 m_projMat;
    glm::mat4 m_viewMat;
    glm::mat4 m_modelMat;
    glm::vec3 m_lightPos;
    QPoint m_lastPos;


    // Application specific members
    std::shared_ptr<Fluid> m_fluid;
    std::shared_ptr<Algae> m_algae;
    std::shared_ptr<Rigid> m_container;
    std::shared_ptr<Rigid> m_staticRigid;
    std::shared_ptr<Rigid> m_activeRigid;
    std::shared_ptr<FluidSystem> m_fluidSystem;
    Mesh m_activeRigidMesh;

    // Rendering
    std::shared_ptr<BioluminescentFluidRenderer> m_bioRenderer;
    std::vector<std::shared_ptr<SphParticleRenderer>> m_sphRenderers;

    QTimer *m_drawTimer;

    // simulation cache
    CacheSystem m_cache;

};

#endif // OPENGLSCENE_H
