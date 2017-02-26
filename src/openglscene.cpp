#include "include/openglscene.h"
#include <iostream>
#include <sys/time.h>
#include <math.h>

#include <QMouseEvent>
#include <QImage>

#include "Mesh/meshloader.h"
#include "MeshSampler/meshsampler.h"


OpenGLScene::OpenGLScene(QWidget *parent) : QOpenGLWidget(parent),
    m_xRot(0),
    m_yRot(0),
    m_zRot(0),
    m_xDis(0),
    m_yDis(0),
    m_zDis(400)
{
    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);

    setFormat(format);

    m_drawTimer = new QTimer(this);
    connect(m_drawTimer, SIGNAL(timeout()), this, SLOT(update()));

    m_simTimer = new QTimer(this);
    connect(m_simTimer, SIGNAL(timeout()), this, SLOT(UpdateSim()));
}


OpenGLScene::~OpenGLScene()
{
    cleanup();
}


QSize OpenGLScene::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize OpenGLScene::sizeHint() const
{
    return QSize(400, 400);
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void OpenGLScene::setXTranslation(int x)
{
    if (x != m_xDis) {
        m_xDis = x;
        emit xTranslationChanged(x);
        update();
    }
}

void OpenGLScene::setYTranslation(int y)
{
    if (y != m_yDis) {
        m_yDis = y;
        emit yTranslationChanged(y);
        update();
    }
}

void OpenGLScene::setZTranslation(int z)
{
    if (z != m_zDis) {
        m_zDis= z;
        emit zTranslationChanged(z);
        update();
    }
}

void OpenGLScene::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_xRot) {
        m_xRot = angle;
        emit xRotationChanged(angle);
        update();
    }
}

void OpenGLScene::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_yRot) {
        m_yRot = angle;
        emit yRotationChanged(angle);
        update();
    }
}

void OpenGLScene::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_zRot) {
        m_zRot = angle;
        emit zRotationChanged(angle);
        update();
    }
}

void OpenGLScene::cleanup()
{
    makeCurrent();
    m_skyboxTex->destroy();
    m_skyboxTex = nullptr;
    doneCurrent();
}



void OpenGLScene::initializeGL()
{
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &OpenGLScene::cleanup);

    glewInit();

    //initializeOpenGLFunctions();
    glClearColor(0.4f, 0.4f, 0.4f, 0.0f);

    //---------------------------------------------------------------------------------------
    // initialise view and projection matrices
    m_viewMat = glm::mat4(1);
    m_viewMat = glm::lookAt(glm::vec3(0,0,0),glm::vec3(0,0,-1),glm::vec3(0,1,0));
    m_projMat = glm::perspective(45.0f, GLfloat(width()) / height(), 0.1f, 1000.0f);

    // Light position is fixed.
    m_lightPos = glm::vec3(0, 0, 70);

    //---------------------------------------------------------------------------------------
    // Create Skybox
    CreateSkybox();

    //---------------------------------------------------------------------------------------

    // fluid
    auto fluidProps = std::shared_ptr<FluidProperty>(new FluidProperty());
    m_fluid = std::shared_ptr<Fluid>(new Fluid(fluidProps, width(), height()));
    m_fluid->SetCubeMap(m_skyboxTex);

    // rigid
    auto fluidSolverProps = std::shared_ptr<FluidSolverProperty>(new FluidSolverProperty());
    auto rigidProps = std::shared_ptr<RigidProperty>(new RigidProperty());

    Mesh boundary = Mesh();
    float dim = 0.95f* fluidSolverProps->gridResolution*fluidSolverProps->gridCellWidth;
    float rad = rigidProps->particleRadius;
    int numRigidAxis = ceil(dim / (rad*2.0f));
    for(int z=0; z<numRigidAxis; z++)
    {
        for(int y=0; y<numRigidAxis; y++)
        {
            for(int x=0; x<numRigidAxis; x++)
            {
                if(x==0 || x==numRigidAxis-1 || y==0 || y==numRigidAxis-1 || z==0 || z==numRigidAxis-1)
                {
                    glm::vec3 pos((x*rad*2.0f)-(dim*0.5f), (y*rad*2.0f)-(dim*0.5f), (z*rad*2.0f)-(dim*0.5f));
                    boundary.verts.push_back(pos);
                }
            }
        }
    }

    rigidProps->numParticles = boundary.verts.size();
    m_container = std::shared_ptr<Rigid>(new Rigid(rigidProps, boundary));


    m_fluidSystem = std::shared_ptr<FluidSystem>(new FluidSystem(fluidSolverProps));
    m_fluidSystem->SetContainer(m_container);
    m_fluidSystem->AddFluid(m_fluid);

    emit FluidInitialised(fluidProps);

    m_fluidSystem->InitialiseSim();


    //---------------------------------------------------------------------------------------


    m_drawTimer->start(16);
    m_simTimer->start(16);

}

void OpenGLScene::paintGL()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    // clean gl window
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // update model matrix
    m_modelMat = glm::mat4(1);
    m_modelMat = glm::translate(m_modelMat, glm::vec3(0.05f*(m_zDis/250.0f)*m_xDis, -0.05f*(m_zDis/250.0f)*m_yDis, -0.1f*m_zDis));
    m_modelMat = glm::rotate(m_modelMat, glm::radians(m_xRot/16.0f), glm::vec3(1,0,0));
    m_modelMat = glm::rotate(m_modelMat, glm::radians(m_yRot/16.0f), glm::vec3(0,1,0));
    glm::mat3 normalMatrix =  glm::inverse(glm::mat3(m_modelMat));
    glm::vec3 camPos = glm::vec3(glm::inverse((m_modelMat)) * glm::vec4(0.0f,0.0f, 0.0f,1.0f));

    //---------------------------------------------------------------------------------------
    // Draw code - replace this with project specific draw stuff

    // Draw skybox first
    DrawSkybox();

    // Draw fluid
    m_fluid->SetShaderUniforms(m_projMat, m_viewMat, m_modelMat, glm::mat4(normalMatrix), m_lightPos, camPos);
    m_fluid->Draw();

//    m_container->SetShaderUniforms(m_projMat, m_viewMat, m_modelMat, glm::mat4(normalMatrix), m_lightPos, camPos);
//    m_container->Draw();

    //---------------------------------------------------------------------------------------

}

void OpenGLScene::UpdateSim()
{
    static double time = 0.0;
    static double t1 = 0.0;
    static double t2 = 0.0;
    struct timeval tim;

    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec/1000000.0);


    m_fluidSystem->StepSimulation();


    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    time += 10*(t2-t1);
//    std::cout<<"fps: "<<1.0/(t2-t1)<<"\n";

}

void OpenGLScene::ResetSim()
{
    m_fluidSystem->ResetSim();
}

void OpenGLScene::resizeGL(int w, int h)
{
    m_fluid->SetFrameSize(w, h);
    m_projMat = glm::perspective(45.0f, GLfloat(w) / h, 0.1f, 1000.0f);
}

void OpenGLScene::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = event->pos();
}

void OpenGLScene::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(m_xRot + 8 * dy);
        setYRotation(m_yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
        setZTranslation(m_zDis + dy);
    }
    else if(event->buttons() & Qt::MiddleButton)
    {
        setXTranslation(m_xDis + dx);
        setYTranslation(m_yDis + dy);
    }
    m_lastPos = event->pos();
}

void OpenGLScene::CreateSkybox()
{
    // create shader
    m_skyboxShader.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shader/Skybox/skybox.vert");
    m_skyboxShader.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shader/Skybox/skybox.frag");
    m_skyboxShader.link();

    // create vertex data
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();
    m_skyboxVAO.create();
    m_skyboxVAO.bind();
    m_skyboxVBO.create();
    m_skyboxVBO.bind();
    const GLfloat cubeVerts[] = {
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };
    m_skyboxVBO.allocate(cubeVerts, 36 * 3 * sizeof(float));
    glFuncs->glEnableVertexAttribArray(m_skyboxShader.attributeLocation("vPos"));
    glFuncs->glVertexAttribPointer(m_skyboxShader.attributeLocation("vPos"), 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
    m_skyboxVBO.release();
    m_skyboxVAO.release();

    // create textures
    m_skyboxTex.reset(new QOpenGLTexture(QOpenGLTexture::TargetCubeMap));
    m_skyboxTex->create();
    m_skyboxTex->bind();
    m_skyboxTex->setSize(2048,2048, 2048);
    m_skyboxTex->setFormat(QOpenGLTexture::RGBAFormat);
    m_skyboxTex->allocateStorage();
    m_skyboxTex->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    m_skyboxTex->setWrapMode(QOpenGLTexture::ClampToEdge);
    m_skyboxTex->setData(0, 0, QOpenGLTexture::CubeMapPositiveX, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, QImage("tex/skybox/right.jpg").mirrored().bits());
    m_skyboxTex->setData(0, 0, QOpenGLTexture::CubeMapNegativeX, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, QImage("tex/skybox/left.jpg").mirrored().bits());
    m_skyboxTex->setData(0, 0, QOpenGLTexture::CubeMapPositiveY, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, QImage("tex/skybox/bottom.jpg").mirrored().bits());
    m_skyboxTex->setData(0, 0, QOpenGLTexture::CubeMapNegativeY, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, QImage("tex/skybox/top.jpg").mirrored().bits());
    m_skyboxTex->setData(0, 0, QOpenGLTexture::CubeMapPositiveZ, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, QImage("tex/skybox/back.jpg").mirrored().bits());
    m_skyboxTex->setData(0, 0, QOpenGLTexture::CubeMapNegativeZ, QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, QImage("tex/skybox/front.jpg").mirrored().bits());
}

void OpenGLScene::DrawSkybox()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_skyboxShader.bind();
    glFuncs->glDepthMask(GL_FALSE);
    glFuncs->glUniformMatrix4fv(m_skyboxShader.uniformLocation("uProjMatrix"), 1, false, &m_projMat[0][0]);
    glm::mat4 mv = glm::mat4(glm::mat3(m_modelMat*m_viewMat));
    glFuncs->glUniformMatrix4fv(m_skyboxShader.uniformLocation("uViewMatrix"), 1, false, &mv[0][0]);
    m_skyboxShader.setUniformValue("uSkyboxTex", 0);
    glFuncs->glActiveTexture(GL_TEXTURE0);
    glFuncs->glBindTexture(GL_TEXTURE_CUBE_MAP, m_skyboxTex->textureId());
    m_skyboxVAO.bind();
    glFuncs->glDrawArrays(GL_TRIANGLES, 0, 36);
    m_skyboxVAO.release();
    glFuncs->glDepthMask(GL_TRUE);
    m_skyboxShader.release();
}
