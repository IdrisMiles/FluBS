#ifndef MAINWINDOW_H
#define MAINWINDOW_H


// Qt includes
#include <QMainWindow>
#include "Widget/fluidpropertywidgetOld.h"
#include "Widget/fluidpropertywidget.h"

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
    std::shared_ptr<FluidPropertyWidgetOld> m_fluidPropertWidget;
    FluidPropertyWidget *m_fpw;

};

#endif // MAINWINDOW_H
