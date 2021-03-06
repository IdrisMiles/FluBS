#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QFileDialog>

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

    connect(ui->scene, &OpenGLScene::SceneFrameRangeChanged, ui->timeline, &TimeLineWidget::OnFrameRangeChanged);


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
        // connect fluid system property changed to openglscene in order ot clear cache
        auto fluidSystem = _fluidSystem.get();
        connect(solverPropWidget, &SolverPropertyWidget::PropertyChanged, [this, fluidSystem](const FluidSolverProperty &_newProperties){
            ui->scene->makeCurrent();
            fluidSystem->SetFluidSolverProperty(_newProperties);
            ui->scene->OnPropertiesChanged();
        });

    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnFluidInitialised(std::shared_ptr<Fluid> _fluid)
{
    if(_fluid != nullptr)
    {
        // create a new fluid property widget
        auto fluidPropWidget = new FluidPropertyWidget(ui->properties, *_fluid->GetProperty());
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


        Fluid* fluid = _fluid.get();
        connect(fluidPropWidget, &FluidPropertyWidget::PropertyChanged, [this, fluidPropWidget, fluid](){
            ui->scene->makeCurrent();
            fluid->SetProperty(fluidPropWidget->GetProperty());
            ui->scene->OnPropertiesChanged();
        });

    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnRigidInitialised(std::shared_ptr<Rigid> _rigid)
{
    if(_rigid != nullptr)
    {
        ui->scene->makeCurrent();

        // create rigid widget
        glm::vec3 pos = _rigid->GetPos();
        glm::vec3 rot = _rigid->GetRot();
        auto rigidPropWidget = new RigidPropertyWidget(ui->properties, *_rigid->GetProperty(), pos.x, pos.y, pos.z, rot.x, rot.y, rot.z);
        std::string name = _rigid->GetName();
        int tabId = ui->properties->addTab(rigidPropWidget, QString(name.c_str()));

        // add rigid to tree view
        auto item = new QTreeWidgetItem();
        item->setText(0,QString(name.c_str()));
        ui->outliner->addTopLevelItem(item);


        connect(ui->outliner, &QTreeWidget::itemClicked, [this, item, tabId, name](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem == item)
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });

        connect(ui->outliner, &QTreeWidget::itemDoubleClicked, [this, _rigid, item, tabId, name](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem == item)
            {
                ui->properties->setCurrentIndex(tabId);

                QMessageBox msgBox;
                msgBox.setText("Do you want to Delete this Rigid Object: "+QString(name.c_str()));
                msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
                msgBox.setDefaultButton(QMessageBox::Cancel);
                if(msgBox.exec() == QMessageBox::Yes)
                {
                    ui->scene->RemoveRigid(_rigid);
                    ui->outliner->removeItemWidget(item, 0);
                    delete item;
                    ui->properties->removeTab(tabId);
                }

            }
        });


        connect(rigidPropWidget, &RigidPropertyWidget::PropertyChanged, [this, rigidPropWidget, _rigid](){
            ui->scene->makeCurrent();
            _rigid->SetProperty(rigidPropWidget->GetProperty());
            // TODO:
            // OpenGLScene to update rigid in fludi solver, update hash pos, volumes, density.
            ui->scene->OnPropertiesChanged();
        });


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
        auto algaePropWidget = new AlgaePropertyWidget(ui->properties, *_algae->GetProperty());
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

        Algae* algae = _algae.get();
        connect(algaePropWidget, &AlgaePropertyWidget::PropertyChanged, [this, algaePropWidget, algae](){
            ui->scene->makeCurrent();
            algae->SetProperty(algaePropWidget->GetProperty());
            ui->scene->OnPropertiesChanged();
        });
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::Save()
{
    // Get filename to save to
    QString fileName = QDir().relativeFilePath(QFileDialog::getSaveFileName(this, tr("Save"), "./", tr("JSON Files (*.json *.jsn)")));
    if(fileName.isEmpty() || fileName.isNull())
    {
        return;
    }

    ui->scene->SaveScene(ui->progressBar, fileName);
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::Open()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Save"), "./", tr("JSON Files (*.json *.jsn)"));
    if(fileName.isEmpty() || fileName.isNull())
    {
        return;
    }

    ClearScene();

    ui->scene->OpenScene(ui->progressBar, fileName);
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::AddRigid(const std::string type)
{
    ui->timeline->Pause();
    ui->scene->AddRigid(ui->progressBar, type);
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::CreateMenus()
{
    m_fileMenu = menuBar()->addMenu(tr("&File"));
    m_fileMenu->addAction(m_saveAction);
    m_fileMenu->addAction(m_openAction);


    m_editMenu = menuBar()->addMenu(tr("&Edit"));
    m_rigidMenu = m_editMenu->addMenu(tr("&Add Rigid Body"));
    m_rigidMenu->addAction(m_addRigidCubeAction);
    m_rigidMenu->addAction(m_addRigidSphereAction);
    m_rigidMenu->addAction(m_addRigidMeshAction);


}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::CreateActions()
{
    m_saveAction = new QAction(tr("&Save"), this);
    m_saveAction->setShortcut(QKeySequence::Save);
    connect(m_saveAction, &QAction::triggered, this, &MainWindow::Save);

    m_openAction = new QAction(tr("&Open"), this);
    m_openAction->setShortcut(QKeySequence::Open);
    connect(m_openAction, &QAction::triggered, this, &MainWindow::Open);


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

void MainWindow::ClearScene()
{
    ui->scene->ClearScene();
    ui->outliner->clear();
    ui->properties->clear();
}

//---------------------------------------------------------------------------------------------------------------------------------------------



//---------------------------------------------------------------------------------------------------------------------------------------------
