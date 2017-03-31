#ifndef SOLVERPROPERTYWIDGET_H
#define SOLVERPROPERTYWIDGET_H

#include <QWidget>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QPushButton>

namespace Ui {
class SolverPropertyWidget;
}

class SolverPropertyWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SolverPropertyWidget(QWidget *parent = 0);
    ~SolverPropertyWidget();

private:
    Ui::SolverPropertyWidget *ui;
};

#endif // SOLVERPROPERTYWIDGET_H
