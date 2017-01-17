#ifndef OPENGLSCENE_H
#define OPENGLSCENE_H


#include <GL/glew.h>

#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QTimer>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "include/fluid.h"
#include "include/meshloader.h"
#include "include/rendermesh.h"


class OpenGLScene : public QOpenGLWidget
{

    Q_OBJECT

public:
    OpenGLScene(QWidget *parent = 0);
    ~OpenGLScene();

    QSize minimumSizeHint() const Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;

    Fluid *GetFluid(const int &_i){_i<m_fluids.size() ? m_fluids[_i] : nullptr;}

    static glm::mat4 getProjMat(){return m_projMat;}
    static glm::mat4 getViewMat(){return m_viewMat;}
    static glm::mat4 getModelMat(){return m_modelMat;}
    static glm::vec3 getLightPos(){return m_lightPos;}

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

signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);
    void xTranslationChanged(int x);
    void yTranslationChanged(int y);
    void zTranslationChanged(int z);
    void FluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty);

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;

private:

    int m_xRot;
    int m_yRot;
    int m_zRot;
    int m_xDis;
    int m_yDis;
    int m_zDis;


    static glm::mat4 m_projMat;
    static glm::mat4 m_viewMat;
    static glm::mat4 m_modelMat;
    static glm::vec3 m_lightPos;
    QPoint m_lastPos;


    // Application specific members
    std::vector<std::shared_ptr<Fluid>> m_fluids;

    QTimer *m_drawTimer;
    QTimer *m_simTimer;


};

#endif // OPENGLSCENE_H
