#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_scene = new OpenGLScene(this);
    ui->gridLayout->addWidget(m_scene, 0, 0, 2, 2);

    m_fluidPropertWidget = std::shared_ptr<FluidPropertyWidget>(new FluidPropertyWidget(this));
    ui->gridLayout->addWidget(m_fluidPropertWidget.get(), 0, 2, 1, 1 );

    connect(m_scene, SIGNAL(FluidInitialised(std::shared_ptr<FluidProperty>)), this, SLOT(NewFluidInitialised(std::shared_ptr<FluidProperty>)));
    connect(m_fluidPropertWidget.get(), SIGNAL(ResetSim()), m_scene, SLOT(ResetSim()));

}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_scene;
}




void MainWindow::NewFluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty)
{
    if(_fluidProperty != nullptr)
    {
        m_fluidPropertWidget->SetFluidProperty(_fluidProperty);
    }
}
