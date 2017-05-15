#include "mainwindow.h"
#include "ui_mainwindow.h"

//---------------------------------------------------------------------------------------------------------------------------------------------

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    //-------------------------------------------------------
    // Widget setup

    // setup widget and grid layout
    ui->setupUi(this);

    ui->gridLayout->addWidget(ui->scene, 0, 0, 2, 4);

    // setup properties tab widgets
    ui->gridLayout->addWidget(ui->propertyGroup, 0, 4, 2, 1 );

    // setup timeline widget
    ui->gridLayout->addWidget(ui->timeline, 2, 0, 1, 5);


    CreateActions();
    CreateMenus();


    //-------------------------------------------------------
    // Connections

    // connect openglscene widget
    connect(ui->scene, SIGNAL(FluidSystemInitialised(std::shared_ptr<FluidSystem>)), this, SLOT(OnFluidSystemInitialised(std::shared_ptr<FluidSystem>)));
    connect(ui->scene, SIGNAL(FluidInitialised(std::shared_ptr<Fluid>)), this, SLOT(OnFluidInitialised(std::shared_ptr<Fluid>)));
    connect(ui->scene, SIGNAL(RigidInitialised(std::shared_ptr<Rigid>)), this, SLOT(OnRigidInitialised(std::shared_ptr<Rigid>)));
    connect(ui->scene, SIGNAL(AlgaeInitialised(std::shared_ptr<Algae>)), this, SLOT(OnAlgaeInitialised(std::shared_ptr<Algae>)));


    connect(ui->timeline, &TimeLineWidget::FrameChanged, ui->scene, &OpenGLScene::OnFrameChanged);

    // connect if caching is set in timeline widget to cachesystem in openglscene
    connect(ui->timeline, &TimeLineWidget::CacheChecked, ui->scene, &OpenGLScene::OnCacheChecked);

    // connect OpenGLScene::FrameFinished with TimeLine
    connect(ui->scene, &OpenGLScene::FrameFinished, ui->timeline, &TimeLineWidget::OnFrameFinished);


}

//---------------------------------------------------------------------------------------------------------------------------------------------

MainWindow::~MainWindow()
{
    delete ui;
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnFluidSystemInitialised(std::shared_ptr<FluidSystem> _fluidSystem)
{
    if(_fluidSystem != nullptr)
    {
        // create property widget
        auto solverPropWidget = new SolverPropertyWidget(ui->properties, _fluidSystem->GetProperty());
        int tabId = ui->properties->addTab(solverPropWidget, "Solver");

        // create outliner item
        auto item = new QTreeWidgetItem();
        QString outlinerObjectName = "Solver";
        item->setText(0,outlinerObjectName);
        ui->outliner->addTopLevelItem(item);


        // connect outliner to properties tab
        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId, outlinerObjectName](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == outlinerObjectName)
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });


        // connect fluid system property changed to fluid system
        auto fluidSystem = _fluidSystem.get();
        connect(solverPropWidget, &SolverPropertyWidget::PropertyChanged, [this, fluidSystem](const FluidSolverProperty &_newProperties){
            fluidSystem->SetFluidSolverProperty(_newProperties);
        });

        // connect fluid system property changed to openglscene in order ot clear cache
        connect(solverPropWidget, &SolverPropertyWidget::PropertyChanged, ui->scene, &OpenGLScene::OnPropertiesChanged);
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnFluidInitialised(std::shared_ptr<Fluid> _fluid)
{
    if(_fluid != nullptr)
    {
        // create a new fluid property widget
        auto fluidPropWidget = new FluidPropertyWidget(ui->properties, _fluid->GetProperty());
        int tabId = ui->properties->addTab(fluidPropWidget, "Fluid");

        // add fluid to outliner
        auto item = new QTreeWidgetItem();
        item->setText(0,"Fluid");
        ui->outliner->addTopLevelItem(item);

        // connect outliner to fluid properties widget
        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == "Fluid")
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });

        // connect fluid system property changed to fluid system
//        auto fluid = _fluid.get();
//        connect(fluidPropWidget, &FluidPropertyWidget::PropertyChanged, [this, fluid](const FluidProperty *_newProperties){
//            fluid->SetFluidProperty(_newProperties);
//        });

        // connect fluid property changed to openglscene in order ot clear cache
        connect(fluidPropWidget, &FluidPropertyWidget::PropertyChanged, ui->scene, &OpenGLScene::OnPropertiesChanged);
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnRigidInitialised(std::shared_ptr<Rigid> _rigid)
{
    if(_rigid != nullptr)
    {
        auto rigidPropWidget = new RigidPropertyWidget(ui->properties, _rigid->GetProperty());
        int tabId = ui->properties->addTab(rigidPropWidget, "Rigid");


        auto item = new QTreeWidgetItem();
        item->setText(0,"Rigid");
        ui->outliner->addTopLevelItem(item);

        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == "Rigid")
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });

        // connect rigid property changed to openglscene in order ot clear cache
        connect(rigidPropWidget, &RigidPropertyWidget::PropertyChanged, ui->scene, &OpenGLScene::OnPropertiesChanged);

        connect(rigidPropWidget, &RigidPropertyWidget::TransformChanged, [this, _rigid](float posX, float posY, float posZ, float rotX, float rotY, float rotZ){
            glm::vec3 pos(posX, posY, posZ);
            glm::vec3 rot(rotX, rotY, rotZ);

            _rigid->UpdateMesh(pos, rot);
        });
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnAlgaeInitialised(std::shared_ptr<Algae> _algae)
{
    if(_algae != nullptr)
    {
        auto algaePropWidget = new AlgaePropertyWidget(ui->properties, _algae->GetProperty());
        int tabId = ui->properties->addTab(algaePropWidget, "Algae");


        auto item = new QTreeWidgetItem();
        item->setText(0,"Algae");
        ui->outliner->addTopLevelItem(item);

        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == "Algae")
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });

        // connect algae property changed to openglscene in order ot clear cache
        connect(algaePropWidget, &AlgaePropertyWidget::PropertyChanged, ui->scene, &OpenGLScene::OnPropertiesChanged);
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::Cache()
{
    ui->scene->CacheOutSimulation(ui->progressBar);
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::Load()
{
    ui->scene->LoadSimulation(ui->progressBar);
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::AddRigid(const std::string type)
{
    ui->scene->AddRigid(ui->progressBar, type);
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::AddRigidCube()
{
    ui->scene->AddRigid(ui->progressBar, "cube");
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::AddRigidSphere()
{
    ui->scene->AddRigid(ui->progressBar, "sphere");
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::AddRigidMesh()
{
    ui->scene->AddRigid(ui->progressBar, "mesh");
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::CreateMenus()
{
    m_fileMenu = menuBar()->addMenu(tr("&File"));
    m_fileMenu->addAction(m_cacheAction);
    m_fileMenu->addAction(m_loadAction);


    m_editMenu = menuBar()->addMenu(tr("&Edit"));
    m_rigidMenu = m_editMenu->addMenu(tr("&Add Rigid Body"));
    m_rigidMenu->addAction(m_addRigidCubeAction);
    m_rigidMenu->addAction(m_addRigidSphereAction);
    m_rigidMenu->addAction(m_addRigidMeshAction);


}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::CreateActions()
{
    m_cacheAction = new QAction(tr("&Cache"), this);
    connect(m_cacheAction, &QAction::triggered, this, &MainWindow::Cache);

    m_loadAction = new QAction(tr("&Load"), this);
    connect(m_loadAction, &QAction::triggered, this, &MainWindow::Load);


    m_addRigidCubeAction = new QAction(tr("&Cube"), this);
    connect(m_addRigidCubeAction, &QAction::triggered, [this](){
        AddRigid("cube");
    });

    m_addRigidSphereAction = new QAction(tr("&Sphere"), this);
    connect(m_addRigidSphereAction, &QAction::triggered, [this](){
        AddRigid("sphere");
    });

    m_addRigidMeshAction = new QAction(tr("&Mesh"), this);
    connect(m_addRigidMeshAction, &QAction::triggered, [this](){
        AddRigid("mesh");
    });
}

//---------------------------------------------------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------------------------------------------------



//---------------------------------------------------------------------------------------------------------------------------------------------
