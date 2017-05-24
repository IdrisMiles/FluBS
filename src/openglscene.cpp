#include "include/openglscene.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <sys/time.h>
#include <math.h>

#include <QMouseEvent>
#include <QImage>
#include <QFileDialog>

#include <glm/gtx/euler_angles.hpp>

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

void OpenGLScene::SaveScene(QProgressBar *progress, QString fileName)
{
    // Get filename to save to
    if(fileName=="" || fileName.isEmpty() || fileName.isNull())
    {
        fileName = QFileDialog::getSaveFileName(this, tr("Save"), "./", tr("JSON Files (*.json *.jsn)"));
        if(fileName.isEmpty() || fileName.isNull())
        {
            return;
        }
    }

    // create json data
    json scene;
    scene["numSolvers"] = 1;
    scene["numFluids"] = 1;
    scene["numAlgaes"] = 1;
    scene["numRigids"] = m_rigids.size();
    scene["solver"] = m_fluidSystem->GetProperty();
    scene["fluid"] = *m_fluid->GetProperty();
    scene["algae"] = *m_algae->GetProperty();
    for(size_t i=0; i<m_rigids.size(); i++)
    {
        scene["rigids"].push_back({{"property", *m_rigids[i]->GetProperty()},
                                   {"type", m_rigids[i]->GetType()},
                                   {"name", m_rigids[i]->GetName()},
                                   {"file", m_rigids[i]->GetFileName()},
                                   {"pos", m_rigids[i]->GetPos()},
                                   {"rot", m_rigids[i]->GetRot()}});
    }

    if(!QDir(fileName).exists())
    {
        QDir().mkdir(fileName);
    }


    // cache sim data
    scene["numCachedFrames"] = m_cache.GetCachedRange();
    scene["cachedFiles"] = fileName.toStdString()+"/sim_";
    m_cache.CacheOutToDisk(fileName.toStdString()+"/sim_", progress);

    // save scene file
    std::string _file = fileName.toStdString()+".json";
    std::ofstream ofs(_file);
    if(ofs.is_open())
    {
        ofs << std::setw(4)<< scene << std::endl;

        ofs.close();
    }


}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::OpenScene(QProgressBar *progress, QString fileName)
{
    if(fileName=="" || fileName.isEmpty() || fileName.isNull())
    {
        std::cout<<fileName.toStdString()<<" | KEHHHH\n";
        fileName = QFileDialog::getOpenFileName(this, tr("Open Scene"), "./", tr("JSON Files (*.json *.jsn)"));
        if(fileName.isEmpty() || fileName.isNull())
        {
            return;
        }
    }

    json scene;
    std::ifstream i(fileName.toStdString());
    i >> scene;

    int numSolver = scene.at("numSolvers").get<int>();
    if(numSolver > 0)
    {
        FluidSolverProperty fluidSolverProps = scene.at("solver").get<FluidSolverProperty>();
//        AddSolver(fluidSolverProps);
        m_fluidSystem->SetFluidSolverProperty(fluidSolverProps);

        emit FluidSystemInitialised(m_fluidSystem);
    }

    int numFluid = scene.at("numFluids").get<int>();
    if(numFluid > 0)
    {
        FluidProperty fluidProps = scene.at("fluid").get<FluidProperty>();
//        auto fp = std::shared_ptr<FluidProperty>(new FluidProperty);// std::make_shared<FluidProperty>(fluidProps);
//        *fp = fluidProps;
//        AddFluid(fp);
        m_fluid->SetProperty(fluidProps);
        emit FluidInitialised(m_fluid);
    }

    int numAlgae = scene.at("numAlgaes").get<int>();
    if(numAlgae > 0)
    {
        AlgaeProperty algaeProps = scene.at("algae").get<AlgaeProperty>();
//        auto ap = std::make_shared<AlgaeProperty>(algaeProps);
//        AddAlgae(ap);
        m_algae->SetProperty(algaeProps);
        emit AlgaeInitialised(m_algae);
    }

    int numRigids = scene.at("numRigids").get<int>();
    for(size_t i=0; i<numRigids; i++)
    {
        RigidProperty rigidProp = scene["rigids"][i].at("property").get<RigidProperty>();
        std::string rigidType = scene["rigids"][i].at("type").get<std::string>();
        std::string rigidName = scene["rigids"][i].at("name").get<std::string>();
        std::string meshFileName = scene["rigids"][i].at("file").get<std::string>();
        glm::vec3 rigidPos = scene["rigids"][i].at("pos").get<glm::vec3>();
        glm::vec3 rigidRot = scene["rigids"][i].at("rot").get<glm::vec3>();

        LoadRigid(progress, rigidType, rigidProp, rigidName, meshFileName, rigidPos, rigidRot);
    }

    m_fluidSystem->InitialiseSim();

//    m_bioRenderer = std::shared_ptr<BioluminescentFluidRenderer>(new BioluminescentFluidRenderer(width(), height()));
//    m_bioRenderer->SetSphParticles(m_fluid, m_algae);

    OnPropertiesChanged();

    int numFrame = scene.at("numCachedFrames").get<int>();
    if(numFrame > 0)
    {
        std::string cachedFileName = scene.at("cachedFiles").get<std::string>();
        if(!cachedFileName.empty())
        {
            std::vector<std::string> files;
            for(int i=0; i<numFrame; i++)
            {
                std::stringstream ss;
                ss << std::setw(4) << std::setfill('0') << i;
                files.push_back(cachedFileName+ss.str()+".json");
            }

            m_cache.LoadCacheFromDisk(files, progress);
        }
    }
}


//------------------------------------------------------------------------------------------------------------

void OpenGLScene::ClearScene()
{

//    RemoveFluid();
//    RemoveAlgae();
//    RemoveFluidSolver();


    // clear rigids
    for(size_t i=0; i<m_rigids.size(); i++)
    {
        RemoveRigid(m_rigids[i]);
    }

}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::AddSolver(FluidSolverProperty fluidSolverProps)
{
    m_fluidSystem.reset(new FluidSystem(fluidSolverProps));
    emit FluidSystemInitialised(m_fluidSystem);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::AddContainer(std::shared_ptr<RigidProperty> containerProps)
{
    // rigid container
    Mesh boundary = Mesh();
    auto fluidSolverProps = m_fluidSystem->GetProperty();
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

    m_fluidSystem->SetContainer(m_container);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::AddFluid(std::shared_ptr<FluidProperty> fluidProps)
{
    m_fluid = std::shared_ptr<Fluid>(new Fluid(fluidProps));
    m_fluidSystem->AddFluid(m_fluid);
    emit FluidInitialised(m_fluid);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::AddAlgae(std::shared_ptr<AlgaeProperty> algaeProps)
{
    m_algae = std::shared_ptr<Algae>(new Algae(algaeProps));
    m_fluidSystem->AddAlgae(m_algae);
    emit AlgaeInitialised(m_algae);
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::RemoveFluidSolver()
{

}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::RemoveFluid()
{
    auto renderIt = m_sphRenderers.cbegin();
    for(;renderIt != m_sphRenderers.cend(); ++renderIt)
    {
        if((*renderIt)->GetSphParticles()->GetName() == m_fluid->GetName())
        {
            m_sphRenderers.erase(renderIt);
            break;
        }
    }
    m_bioRenderer = nullptr;
    m_fluidSystem->RemoveFluid(m_fluid);
    m_fluid = nullptr;
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::RemoveAlgae()
{
    auto renderIt = m_sphRenderers.cbegin();
    for(;renderIt != m_sphRenderers.cend(); ++renderIt)
    {
        if((*renderIt)->GetSphParticles()->GetName() == m_algae->GetName())
        {
            m_sphRenderers.erase(renderIt);
            break;
        }
    }
    m_bioRenderer = nullptr;
    m_fluidSystem->RemoveAlgae(m_algae);
    m_algae = nullptr;
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::RemoveRigid(std::shared_ptr<Rigid> rigid)
{

    auto renderIt = m_sphRenderers.cbegin();
    for(;renderIt != m_sphRenderers.cend(); ++renderIt)
    {
        if((*renderIt)->GetSphParticles()->GetName() == rigid->GetName())
        {
            m_sphRenderers.erase(renderIt);
            break;
        }
    }

    m_fluidSystem->RemoveRigid(rigid);


    auto rit = m_rigids.cbegin();
    for(;rit != m_rigids.cend(); ++rit)
    {
        if(*rit == rigid)
        {
            m_rigids.erase(rit);
            break;
        }
    }
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::AddRigid(QProgressBar *progress, std::string type)
{
    int progressCount = 0;
    progress->setMaximum(4);
    progress->setValue(progressCount++);

    makeCurrent();
    std::shared_ptr<Rigid> rigid;

    progress->setValue(progressCount++);
    static int cubeCount=0;
    static int sphereCount=0;
    static int meshCount=0;
    std::string name;

    //---------------------------------------------------------------
    // create rigid
    if(type == "cube")
    {
        rigid = CreateRigidCube();
        name = "cube"+std::to_string(cubeCount++);
    }
    else if(type == "sphere")
    {
        rigid = CreateRigidSphere();
        name = "sphere"+std::to_string(sphereCount++);
        rigid->SetType(type);
    }
    else if(type == "mesh")
    {
        QString qFileName = QFileDialog::getOpenFileName(this, tr("Load Mesh"), "./", tr("Mesh Files (*.obj *.dae)"));
        if(qFileName.isEmpty() || qFileName.isNull())
        {
            std::cout<<"No file selected. Loading default cube\n";
            return;
        }
        rigid = CreateRigidMesh(qFileName.toStdString());
        name  = "mesh"+std::to_string(meshCount++);
        rigid->SetFileName(qFileName.toStdString());
    }
    else
    {
        progress->setValue(0);
        return;
    }

    rigid->SetName(name);
    rigid->SetType(type);
    progress->setValue(progressCount++);

    //---------------------------------------------------------------
    // Add rigid to solver
    m_rigids.push_back(rigid);
    m_fluidSystem->AddRigid(rigid);
    emit RigidInitialised(rigid);

    progress->setValue(progressCount++);


    //---------------------------------------------------------------
    // Add rigid to renderer
    m_sphRenderers.push_back(std::shared_ptr<SphParticleRenderer>(new SphParticleRenderer()));
    m_sphRenderers.back()->SetSphParticles(rigid);
    m_sphRenderers.back()->SetColour(glm::vec3(0.4f, 0.4f, 0.4f));

    progress->setValue(progressCount++);

    OnPropertiesChanged();
}

//------------------------------------------------------------------------------------------------------------

void OpenGLScene::LoadRigid(QProgressBar *progress, std::string type, RigidProperty property,
                            std::string name, std::string file,
                            glm::vec3 pos, glm::vec3 rot, glm::vec3 scale)
{
    int progressCount = 0;
    progress->setMaximum(4);
    progress->setValue(progressCount++);

    makeCurrent();
    std::shared_ptr<Rigid> rigid;

    progress->setValue(progressCount++);

    //---------------------------------------------------------------
    // create rigid
    if(type == "cube")
    {
        rigid = CreateRigidCube(property, pos, rot, scale);
    }
    else if(type == "sphere")
    {
        rigid = CreateRigidSphere(property, pos, rot, scale);
    }
    else if(type == "mesh")
    {
        rigid = CreateRigidMesh(file, property, pos, rot, scale);
    }
    else
    {
        progress->setValue(0);
        return;
    }

    rigid->SetName(name);
    rigid->SetType(type);
    rigid->SetFileName(file);

    progress->setValue(progressCount++);

    //---------------------------------------------------------------
    // Add rigid to solver
    m_rigids.push_back(rigid);
    m_fluidSystem->AddRigid(rigid);
    emit RigidInitialised(rigid);

    progress->setValue(progressCount++);


    //---------------------------------------------------------------
    // Add rigid to renderer
    m_sphRenderers.push_back(std::shared_ptr<SphParticleRenderer>(new SphParticleRenderer()));
    m_sphRenderers.back()->SetSphParticles(rigid);
    m_sphRenderers.back()->SetColour(glm::vec3(0.4f, 0.4f, 0.4f));

    progress->setValue(progressCount++);
    OnPropertiesChanged();
}

//------------------------------------------------------------------------------------------------------------

std::shared_ptr<Rigid> OpenGLScene::CreateRigidCube(RigidProperty property, glm::vec3 pos, glm::vec3 rot, glm::vec3 scale)
{
    auto rigidProps = std::shared_ptr<RigidProperty>(new RigidProperty());
    *rigidProps = property;

    Mesh rigidCubeMesh = Mesh();
    float dim = 1.0f;
    float rad = rigidProps->particleRadius;
    int numRigidAxis = ceil(dim / (rad*2.0f));

    // cube
    for(int z=0; z<numRigidAxis; z++)
    {
        for(int y=0; y<numRigidAxis; y++)
        {
            for(int x=0; x<numRigidAxis; x++)
            {
                if(x==0 || x==numRigidAxis-1 || y==0 || y==numRigidAxis-1 || z==0 || z==numRigidAxis-1)
                {
                    glm::vec3 vert = glm::vec3((x*rad*2.0f)-(dim*0.5f), (y*rad*2.0f)-(dim*0.5f), (z*rad*2.0f)-(dim*0.5f));
                    glm::mat3 t = glm::orientate3(rot);
                    vert = (t*vert)+pos;
                    rigidCubeMesh.verts.push_back(vert);

                }
            }
        }
    }

    rigidProps->numParticles = rigidCubeMesh.verts.size();

    return std::shared_ptr<Rigid>(new Rigid(rigidProps, rigidCubeMesh, "cube"));
}

//------------------------------------------------------------------------------------------------------------

std::shared_ptr<Rigid> OpenGLScene::CreateRigidSphere(RigidProperty property, glm::vec3 pos, glm::vec3 rot, glm::vec3 scale)
{
    auto rigidProps = std::shared_ptr<RigidProperty>(new RigidProperty());
    *rigidProps = property;

    Mesh rigidSphereMesh = Mesh();
    int _stacks = 15;
    int _slices = 40;
    float _radius = 0.5f;

    //sphere
    for( int t = 1 ; t < _stacks-1 ; t++ )
    {
        float theta1 = ( (float)(t)/(_stacks-1) )*glm::pi<float>();

        for( int p = 0 ; p < _slices ; p++ )
        {
            float phi1 = ( (float)(p)/(_slices-1) )*2*glm::pi<float>();

            glm::vec3 vert = _radius * glm::vec3(sin(theta1)*cos(phi1), cos(theta1), -sin(theta1)*sin(phi1));
            glm::mat3 t = glm::orientate3(rot);
            vert = (t*vert)+pos;
            rigidSphereMesh.verts.push_back(vert);
        }
    }
    rigidSphereMesh.verts.push_back(pos + (_radius * glm::vec3(0.0f, 1.0f, 0.0f)));
    rigidSphereMesh.verts.push_back(pos + (_radius * glm::vec3(0.0f, -1.0f, 0.0f)));

    rigidProps->numParticles = rigidSphereMesh.verts.size();
    return std::shared_ptr<Rigid>(new Rigid(rigidProps, rigidSphereMesh, "sphere"));
}

//------------------------------------------------------------------------------------------------------------

std::shared_ptr<Rigid> OpenGLScene::CreateRigidMesh(std::string meshFile, RigidProperty property, glm::vec3 pos, glm::vec3 rot, glm::vec3 scale)
{
    auto rigidProps = std::shared_ptr<RigidProperty>(new RigidProperty());
    *rigidProps = property;
    Mesh rigidMesh = Mesh();

    auto meshes = MeshLoader::LoadMesh(meshFile);
    if(meshes.size() < 1)
    {
        std::cout<<"No meshes in file. Loading default cube\n";
        return CreateRigidCube();
    }

    rigidMesh = MeshSampler::BaryCoord::SampleMesh(meshes[0], 1000);


    rigidProps->numParticles = rigidMesh.verts.size();
    std::shared_ptr<Rigid> rigid = std::shared_ptr<Rigid>(new Rigid(rigidProps, rigidMesh, meshFile));
    rigid->UpdateMesh(pos, rot);
    return rigid;
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

    AddSolver();
    AddContainer();
    AddFluid();
    AddAlgae(std::shared_ptr<AlgaeProperty>(new AlgaeProperty(200.0f, 1.0f, 1.0f, 64000, 1.0f, 0.1f, 998.36f)));


    m_fluidSystem->InitialiseSim();


    //---------------------------------------------------------------------------------------
    // sph renderers
    m_bioRenderer = std::shared_ptr<BioluminescentFluidRenderer>(new BioluminescentFluidRenderer(width(), height()));
    m_bioRenderer->SetCubeMap(m_skyboxTex);
    m_bioRenderer->SetSphParticles(m_fluid, m_algae);


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

            if(frame == 0)
            {
                m_fluidSystem->InitialiseSim();
            }
            else
            {
                m_fluidSystem->StepSim();
            }
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
    m_fluidSystem->ResetSim();
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
