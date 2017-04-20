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
    connect(ui->scene, SIGNAL(FluidSystemInitialised(std::shared_ptr<FluidSolverProperty>)), this, SLOT(OnFluidSystemInitialised(std::shared_ptr<FluidSolverProperty>)));
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

void MainWindow::OnFluidSystemInitialised(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty)
{
    if(_fluidSolverProperty != nullptr)
    {
        auto solverPropWidget = new SolverPropertyWidget(ui->properties, _fluidSolverProperty);
        int tabId = ui->properties->addTab(solverPropWidget, "Solver");
        solverPropWidget->SetProperty(_fluidSolverProperty);

        auto item = new QTreeWidgetItem();
        item->setText(0,"Solver");
        ui->outliner->addTopLevelItem(item);

        connect(ui->outliner, &QTreeWidget::itemClicked, ui->properties, [this, tabId](QTreeWidgetItem* clickedItem, int column){
            if(clickedItem->text(column) == "Solver")
            {
                ui->properties->setCurrentIndex(tabId);
            }
        });
    }
}

//---------------------------------------------------------------------------------------------------------------------------------------------

void MainWindow::OnFluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty)
{
    if(_fluidProperty != nullptr)
    {
        auto fluidPropWidget = new FluidPropertyWidget(ui->properties, _fluidProperty);
        int tabId = ui->properties->addTab(fluidPropWidget, "Fluid");
        fluidPropWidget->SetProperty(_fluidProperty);

        auto item = new QTreeWidgetItem();
        item->setText(0,"Fluid");
        ui->outliner->addTopLevelItem(item);

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
