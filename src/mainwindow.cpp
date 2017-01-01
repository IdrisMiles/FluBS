#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_scene = new OpenGLScene(this);
    ui->gridLayout->addWidget(m_scene, 0, 0, 2, 2);

    m_fluidPropertWidegt = new FluidPropertyWidget(this);
    ui->gridLayout->addWidget(m_fluidPropertWidegt, 0, 2, 1, 1 );


}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_scene;
}

