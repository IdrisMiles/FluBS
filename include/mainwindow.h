#ifndef MAINWINDOW_H
#define MAINWINDOW_H


// Qt includes
#include <QMainWindow>
#include "fluidpropertywidget.h"
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


private:
    Ui::MainWindow *ui;
    OpenGLScene *m_scene;
    FluidPropertyWidget * m_fluidPropertWidegt;
};

#endif // MAINWINDOW_H
