#ifndef MAINWINDOW_H
#define MAINWINDOW_H


// Qt includes
#include <QMainWindow>
#include "Widget/solverpropertywidget.h"
#include "Widget/fluidpropertywidget.h"
#include "Widget/algaepropertywidget.h"
#include "Widget/rigidpropertywidget.h"

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
    void OnFluidSystemInitialised(std::shared_ptr<FluidSolverProperty> _fluidSolverProperty);
    void OnFluidInitialised(std::shared_ptr<FluidProperty> _fluidProperty);
    void OnRigidInitialised(std::shared_ptr<RigidProperty> _rigidProperty);
    void OnAlgaeInitialised(std::shared_ptr<AlgaeProperty> _algaeProperty);

private:
    Ui::MainWindow *ui;

};

#endif // MAINWINDOW_H
