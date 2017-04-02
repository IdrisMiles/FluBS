#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    // setup widget and grid layout
    ui->setupUi(this);
    ui->gridLayout->addWidget(ui->scene, 0, 0, 2, 2);

    // setup timeline widget
    ui->gridLayout->addWidget(ui->timeline, 2, 0, 1, 3);

    // setup property widgets
    m_fluidPropertWidget = std::shared_ptr<FluidPropertyWidgetOld>(new FluidPropertyWidgetOld(this));
    m_fpw = new FluidPropertyWidget(this);
    m_apw = new AlgaePropertyWidget(this);
    m_rpw = new RigidPropertyWidget(this);

    // setup properties tab widgets
    ui->properties->addTab(m_fluidPropertWidget.get(), "Fluid");
    ui->properties->addTab(m_fpw, "Fluid new");
    ui->properties->addTab(m_apw, "Algae");
    ui->properties->addTab(m_rpw, "Rigid");
    ui->gridLayout->addWidget(ui->properties, 0, 2, 2, 1 );


    // setup openglscene widget
    connect(ui->scene, SIGNAL(FluidInitialised(std::shared_ptr<FluidProperty>)), this, SLOT(NewFluidInitialised(std::shared_ptr<FluidProperty>)));
    connect(m_fluidPropertWidget.get(), SIGNAL(ResetSim()), ui->scene, SLOT(ResetSim()));
    connect(ui->timeline, &TimeLineWidget::FrameChanged, ui->scene, &OpenGLScene::UpdateSim);

}

MainWindow::~MainWindow()
{
    delete ui;
}




void MainWindow::NewFluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty)
{
    if(_fluidProperty != nullptr)
    {
        m_fluidPropertWidget->SetFluidProperty(_fluidProperty);
    }
}
