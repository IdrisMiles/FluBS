#include "mainwindow.h"
#include "ui_mainwindow.h"

//---------------------------------------------------------------------------------------------------------------------------------------------

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    // setup widget and grid layout
    ui->setupUi(this);
    ui->gridLayout->addWidget(ui->scene, 0, 0, 2, 2);

    // setup timeline widget
    ui->gridLayout->addWidget(ui->timeline, 2, 0, 1, 3);

    // setup properties tab widgets
    ui->gridLayout->addWidget(ui->propertyGroup, 0, 2, 2, 1 );


    // setup openglscene widget
    connect(ui->scene, SIGNAL(FluidSystemInitialised(std::shared_ptr<FluidSystem>)), this, SLOT(OnFluidSystemInitialised(std::shared_ptr<FluidSystem>)));
    connect(ui->scene, SIGNAL(FluidInitialised(std::shared_ptr<FluidProperty>)), this, SLOT(OnFluidInitialised(std::shared_ptr<FluidProperty>)));
    connect(ui->scene, SIGNAL(RigidInitialised(std::shared_ptr<RigidProperty>)), this, SLOT(OnRigidInitialised(std::shared_ptr<RigidProperty>)));
    connect(ui->scene, SIGNAL(AlgaeInitialised(std::shared_ptr<AlgaeProperty>)), this, SLOT(OnAlgaeInitialised(std::shared_ptr<AlgaeProperty>)));

    connect(ui->timeline, &TimeLineWidget::FrameChanged, ui->scene, &OpenGLScene::OnFrameChanged);

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
        auto solverPropWidget = new SolverPropertyWidget(ui->properties, _fluidSystem->GetProperty());
        int tabId = ui->properties->addTab(solverPropWidget, "Solver");


        auto item = new QTreeWidgetItem();
        QString outlinerObjectName = "Solver";
        item->setText(0,outlinerObjectName);
        ui->outliner->addTopLevelItem(item);


        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId, outlinerObjectName](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == outlinerObjectName)
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });


        auto fluidSystem = _fluidSystem.get();
        connect(solverPropWidget, &SolverPropertyWidget::PropertyChanged, [this, fluidSystem](const FluidSolverProperty &_newProperties){
            std::cout<<"mainWindow connection solver props changed\n";
            fluidSystem->SetFluidSolverProperty(_newProperties);
        });
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnFluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty)
{
    if(_fluidProperty != nullptr)
    {
        // create a new fluid property widget
        auto fluidPropWidget = new FluidPropertyWidget(ui->properties, _fluidProperty);
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
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnRigidInitialised(std::shared_ptr<RigidProperty> _rigidProperty)
{
    if(_rigidProperty != nullptr)
    {
        auto rigidPropWidget = new RigidPropertyWidget(ui->properties, _rigidProperty);
        int tabId = ui->properties->addTab(rigidPropWidget, "Rigid");
        rigidPropWidget->SetProperty(_rigidProperty);

        auto item = new QTreeWidgetItem();
        item->setText(0,"Rigid");
        ui->outliner->addTopLevelItem(item);

        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == "Rigid")
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnAlgaeInitialised(std::shared_ptr<AlgaeProperty> _algaeProperty)
{
    if(_algaeProperty != nullptr)
    {
        auto algaePropWidget = new AlgaePropertyWidget(ui->properties, _algaeProperty);
        int tabId = ui->properties->addTab(algaePropWidget, "Algae");
        algaePropWidget->SetProperty(_algaeProperty);

        auto item = new QTreeWidgetItem();
        item->setText(0,"Algae");
        ui->outliner->addTopLevelItem(item);

        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == "Algae")
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------
