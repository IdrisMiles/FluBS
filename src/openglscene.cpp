#include "include/openglscene.h"
#include <iostream>
#include <sys/time.h>
#include <math.h>

#include <QMouseEvent>
#include <QImage>
#include <QFileDialog>

#include "Mesh/meshloader.h"
#include "MeshSampler/meshsampler.h"


//------------------------------------------------------------------------------------------------------------

OpenGLScene::OpenGLScene(QWidget *parent) : QOpenGLWidget(parent),
    m_xRot(0),
    m_yRot(0),
    m_zRot(0),
    m_xDis(0),
    m_yDis(0),
    m_zDis(400),
    m_isCaching(true)
{
    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);

    setFormat(format);

    m_drawTimer = new QTimer(this);
    connect(m_drawTimer, SIGNAL(timeout()), this, SLOT(update()));

}


//------------------------------------------------------------------------------------------------------------


OpenGLScene::~OpenGLScene()
{
//    m_cache.WriteCache();
    cleanup();
}


//------------------------------------------------------------------------------------------------------------


QSize OpenGLScene::minimumSizeHint() const
{
    return QSize(50, 50);
}

//------------------------------------------------------------------------------------------------------------

QSize OpenGLScene::sizeHint() const
{
    return QSize(400, 400);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::OnCacheOutSimulation(QProgressBar *progress)
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Cache Out"), "./", tr("JSON Files (*.json *.jsn)"));
    if(fileName.isEmpty() || fileName.isNull())
    {
        return;
    }
    
    m_cache.CacheOutToDisk(fileName.toStdString(), progress);

}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::OnLoadSimulation(QProgressBar *progress)
{
    std::vector<QString> qFileNames = QFileDialog::getOpenFileNames(this, tr("Cache Out"), "./", tr("JSON Files (*.json *.jsn)")).toVector().toStdVector();
    if(qFileNames.empty())
    {
        return;
    }

    std::vector<std::string> fileNames;
    for(auto &f : qFileNames)
    {
        fileNames.push_back(f.toStdString());
    }

    m_cache.LoadCacheFromDisk(fileNames, progress);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::OnSetFrameRange(int start, int end)
{

}

//------------------------------------------------------------------------------------------------------------

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::setXTranslation(int x)
{
    if (x != m_xDis) {
        m_xDis = x;
        emit xTranslationChanged(x);
        update();
    }
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::setYTranslation(int y)
{
    if (y != m_yDis) {
        m_yDis = y;
        emit yTranslationChanged(y);
        update();
    }
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::setZTranslation(int z)
{
    if (z != m_zDis) {
        m_zDis= z;
        emit zTranslationChanged(z);
        update();
    }
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_xRot) {
        m_xRot = angle;
        emit xRotationChanged(angle);
        update();
    }
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_yRot) {
        m_yRot = angle;
        emit yRotationChanged(angle);
        update();
    }
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_zRot) {
        m_zRot = angle;
        emit zRotationChanged(angle);
        update();
    }
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::cleanup()
{
    makeCurrent();
    m_skyboxTex->destroy();
    m_skyboxTex = nullptr;
    doneCurrent();
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::initializeGL()
{
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &OpenGLScene::cleanup);

    glewInit();
    glClearColor(0.4f, 0.4f, 0.4f, 0.0f);


    //---------------------------------------------------------------------------------------
    // initialise view and projection matrices
    m_viewMat = glm::mat4(1);
    m_viewMat = glm::lookAt(glm::vec3(0,0,0),glm::vec3(0,0,-1),glm::vec3(0,1,0));
    m_projMat = glm::perspective(45.0f, GLfloat(width()) / height(), 0.1f, 1000.0f);

    // Light position is fixed.
    m_lightPos = glm::vec3(30, 100, 70);


    //---------------------------------------------------------------------------------------
    // Create Skybox
    CreateSkybox();





    //---------------------------------------------------------------------------------------
    // Set up simulation here

    auto fluidProps = std::shared_ptr<FluidProperty>(new FluidProperty());
    auto algaeProps = std::shared_ptr<AlgaeProperty>(new AlgaeProperty(64000, 1.0f, 0.1f, 998.36f));
    FluidSolverProperty fluidSolverProps;
    auto containerProps = std::shared_ptr<RigidProperty>(new RigidProperty());
    auto staticRigidProps = std::shared_ptr<RigidProperty>(new RigidProperty());

    // fluid
    m_fluid = std::shared_ptr<Fluid>(new Fluid(fluidProps));
    m_algae = std::shared_ptr<Algae>(new Algae(algaeProps));


    // rigid container
    Mesh boundary = Mesh();
    float dim = 0.95f* fluidSolverProps.gridResolution*fluidSolverProps.gridCellWidth;
    float rad = containerProps->particleRadius;
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
    containerProps->numParticles = boundary.verts.size();
    m_container = std::shared_ptr<Rigid>(new Rigid(containerProps, boundary));


    // rigid static
    Mesh staticRigidMesh = Mesh();
    dim = 0.95f* fluidSolverProps.gridResolution*fluidSolverProps.gridCellWidth;
    rad = staticRigidProps->particleRadius;
    numRigidAxis = ceil((dim*0.1) / (rad*2.0f));
    // cube
    for(int z=0; z<numRigidAxis; z++)
    {
        for(int y=0; y<numRigidAxis; y++)
        {
            for(int x=0; x<numRigidAxis; x++)
            {
                if(x==0 || x==numRigidAxis-1 || y==0 || y==numRigidAxis-1 || z==0 || z==numRigidAxis-1)
                {
                    glm::vec3 pos((x*rad*2.0f)-(dim*0.1f*0.5f), (y*rad*2.0f)-(dim*0.1f*0.5f), (z*rad*2.0f)-(dim*0.1f*0.5f));

                    staticRigidMesh.verts.push_back(pos + glm::vec3(dim*0.2f, -dim*0.4f, dim*0.2f));

                    staticRigidMesh.verts.push_back(pos + glm::vec3(dim*0.2f, -dim*0.4f, -dim*0.2f));

                    staticRigidMesh.verts.push_back(pos + glm::vec3(-dim*0.2f, -dim*0.4f, dim*0.2f));

                    staticRigidMesh.verts.push_back(pos + glm::vec3(-dim*0.2f, -dim*0.4f, -dim*0.2f));
                }
            }
        }
    }

    // sphere
    int _stacks = 15;
    int _slices = 40;
    float _radius = 2.0f;
    glm::vec3 _pos(dim*0.0f, -dim*0.4f, -dim*0.0f);
    for( int t = 1 ; t < _stacks-1 ; t++ )
    {
        float theta1 = ( (float)(t)/(_stacks-1) )*glm::pi<float>();

        for( int p = 0 ; p < _slices ; p++ )
        {
            float phi1 = ( (float)(p)/(_slices-1) )*2*glm::pi<float>();

            glm::vec3 vert = glm::vec3(sin(theta1)*cos(phi1), cos(theta1), -sin(theta1)*sin(phi1));
            staticRigidMesh.verts.push_back(_pos + (_radius*vert));
        }
    }
    staticRigidMesh.verts.push_back(_pos + (_radius * glm::vec3(0.0f, 1.0f, 0.0f)));
    staticRigidMesh.verts.push_back(_pos + (_radius * glm::vec3(0.0f, -1.0f, 0.0f)));

    staticRigidProps->numParticles = staticRigidMesh.verts.size();
    m_staticRigid = std::shared_ptr<Rigid>(new Rigid(staticRigidProps, staticRigidMesh));




    // Fluid system
    m_fluidSystem = std::shared_ptr<FluidSystem>(new FluidSystem(fluidSolverProps));
    m_fluidSystem->SetContainer(m_container);
    m_fluidSystem->AddFluid(m_fluid);
    m_fluidSystem->AddRigid(m_staticRigid);
    m_fluidSystem->AddAlgae(m_algae);

    emit FluidSystemInitialised(m_fluidSystem);
    emit FluidInitialised(m_fluid);
    emit RigidInitialised(m_staticRigid);
    emit AlgaeInitialised(m_algae);

    m_fluidSystem->InitialiseSim();


    //---------------------------------------------------------------------------------------
    // sph renderers
    m_bioRenderer = std::shared_ptr<BioluminescentFluidRenderer>(new BioluminescentFluidRenderer(width(), height()));
    m_bioRenderer->SetCubeMap(m_skyboxTex);
    m_bioRenderer->SetSphParticles(m_fluid, m_algae);



    m_sphRenderers.push_back(std::shared_ptr<SphParticleRenderer>(new SphParticleRenderer()));
    m_sphRenderers.back()->SetSphParticles(m_staticRigid);
    m_sphRenderers.back()->SetColour(glm::vec3(0.4f, 0.4f, 0.4f));

//    m_sphRenderers.push_back(std::shared_ptr<SphParticleRenderer>(new SphParticleRenderer()));
//    m_sphRenderers.back()->SetSphParticles(m_fluid);
//    m_sphRenderers.back()->SetColour(glm::vec3(0.2f, 0.4f, 1.0f));

//    m_sphRenderers.push_back(std::shared_ptr<SphParticleRenderer>(new SphParticleRenderer()));
//    m_sphRenderers.back()->SetSphParticles(m_algae);
//    m_sphRenderers.back()->SetColour(glm::vec3(0.2f, 1.0f, 0.3f));

    //---------------------------------------------------------------------------------------
    // Start simulation and drawing rimers
    m_drawTimer->start(16);

}

//------------------------------------------------------------------------------------------------------------

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
    glm::mat3 normalMatrix =  glm::mat3(m_modelMat);
    glm::vec3 camPos = glm::vec3(glm::inverse((m_modelMat)) * glm::vec4(0.0f,0.0f, 0.0f,1.0f));

    //---------------------------------------------------------------------------------------

    // Draw skybox first
    DrawSkybox();


    for(auto &&sr: m_sphRenderers)
    {
        sr->SetShaderUniforms(m_projMat, m_viewMat, m_modelMat, normalMatrix, m_lightPos, camPos);
        sr->Draw();
    }

    m_bioRenderer->SetShaderUniforms(m_projMat, m_viewMat, m_modelMat, normalMatrix, m_lightPos, camPos);
    m_bioRenderer->Draw();


    //---------------------------------------------------------------------------------------

}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::OnFrameChanged(int frame)
{
    if (frame < 0)
    {
        return;
    }

    if(m_isCaching)
    {
        if(m_cache.IsFrameCached(frame))
        {
            m_cache.Load(frame, m_fluidSystem);
            emit FrameLoaded(frame);
        }
        else
        {
            // check previous frame was cached, if not then need to sim previous frame
            if(!m_cache.IsFrameCached(frame-1))
            {
                OnFrameChanged(frame-1);
            }

            m_fluidSystem->StepSim();
            emit FrameSimmed(frame);

            m_cache.Cache(frame, m_fluidSystem);
            emit FrameCached(frame);
        }
    }
    else
    {
        m_fluidSystem->StepSim();
        emit FrameSimmed(frame);
    }

    emit FrameFinished(frame);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::OnPropertiesChanged()
{
    m_cache.ClearCache(-1);
}

//------------------------------------------------------------------------------------------------------------


void OpenGLScene::OnCacheChecked(bool checked)
{
    m_isCaching = checked;
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::UpdateSim()
{
    m_fluidSystem->StepSim();
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::ResetSim()
{
    m_fluidSystem->ResetSim();
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::resizeGL(int w, int h)
{
    m_bioRenderer->SetFrameSize(w, h);
    m_projMat = glm::perspective(45.0f, GLfloat(w) / h, 0.1f, 1000.0f);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = event->pos();
}

//------------------------------------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::DrawSkybox()
{
    QOpenGLFunctions *glFuncs = QOpenGLContext::currentContext()->functions();

    m_skyboxShader.bind();
    glFuncs->glDepthMask(GL_FALSE);
    glFuncs->glUniformMatrix4fv(m_skyboxShader.uniformLocation("uProjMatrix"), 1, false, &m_projMat[0][0]);
    glm::mat4 mv = glm::mat4(glm::mat3(m_viewMat*m_modelMat));
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

//------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------
