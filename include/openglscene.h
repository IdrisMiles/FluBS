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
#include <QProgressBar>

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

    void CacheOutSimulation(QProgressBar *progress);
    void SaveSimulation(QProgressBar *progress);
    void LoadSimulation(QProgressBar *progress);

    void AddRigid(QProgressBar *progress = nullptr, std::string type = "cube");
    void RemoveRigid(std::shared_ptr<Rigid> rigid);
    void RemoveRigid(std::string name);
    void SaveRigid();
    void LoadRigid();

    void AddFluid(QProgressBar *progress = nullptr);
    void RemoveFluid(std::shared_ptr<Fluid> fluid);
    void SaveFluid();
    void LoadFluid();

    void AddAlgae(QProgressBar *progress = nullptr);
    void RemoveAlgae(std::shared_ptr<Algae> algae);
    void SaveAlgae();
    void LoadAlgae();


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
    void OnPropertiesChanged();
    void OnCacheChecked(bool checked);
    void OnSetFrameRange(int start, int end);

signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);
    void xTranslationChanged(int x);
    void yTranslationChanged(int y);
    void zTranslationChanged(int z);
    void FluidSystemInitialised(std::shared_ptr<FluidSystem> _fluidSolver);
    void FluidInitialised(std::shared_ptr<Fluid> _fluid);
    void RigidInitialised(std::shared_ptr<Rigid> _rigid);
    void AlgaeInitialised(std::shared_ptr<Algae> _algar);
    void FrameCached(int frame);
    void FrameSimmed(int frame);
    void FrameLoaded(int frame);
    void FrameFinished(int frame);
    void CacheCleared();

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;

private:

    std::shared_ptr<Rigid> CreateRigidCube(RigidProperty property = RigidProperty());
    std::shared_ptr<Rigid> CreateRigidSphere(RigidProperty property = RigidProperty());
    std::shared_ptr<Rigid> CreateRigidMesh(RigidProperty property = RigidProperty());

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
    std::vector<std::shared_ptr<Rigid>> m_rigids;
    std::shared_ptr<Rigid> m_container;
    std::shared_ptr<FluidSystem> m_fluidSystem;

    // Rendering
    std::shared_ptr<BioluminescentFluidRenderer> m_bioRenderer;
    std::vector<std::shared_ptr<SphParticleRenderer>> m_sphRenderers;
    QTimer *m_drawTimer;

    // simulation cache
    CacheSystem m_cache;
    bool m_isCaching;

};

#endif // OPENGLSCENE_H
