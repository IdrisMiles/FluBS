#ifndef MAINWINDOW_H
#define MAINWINDOW_H


// Qt includes
#include <QMainWindow>
#include "Widget/fluidpropertywidget.h"
#include "openglscene.h"



namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void NewFluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty);

private:
    Ui::MainWindow *ui;
    OpenGLScene *m_scene;
    std::shared_ptr<FluidPropertyWidget> m_fluidPropertWidget;
};

#endif // MAINWINDOW_H
