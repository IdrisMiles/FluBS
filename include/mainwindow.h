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
    void OnFluidSystemInitialised(std::shared_ptr<FluidSystem> _fluidSystem);
    void OnFluidInitialised(std::shared_ptr<Fluid> _fluid);
    void OnRigidInitialised(std::shared_ptr<Rigid> _rigid);
    void OnAlgaeInitialised(std::shared_ptr<Algae> _algae);

    void Save();
    void Open();
    void AddRigid(const std::string type);

private:
    void CreateMenus();
    void CreateActions();
    void ClearScene();

    Ui::MainWindow *ui;

    QMenu *m_fileMenu;
    QAction *m_saveAction;
    QAction *m_openAction;

    QMenu *m_editMenu;
    QMenu *m_rigidMenu;
    QAction *m_addRigidCubeAction;
    QAction *m_addRigidSphereAction;
    QAction *m_addRigidMeshAction;

};

#endif // MAINWINDOW_H
