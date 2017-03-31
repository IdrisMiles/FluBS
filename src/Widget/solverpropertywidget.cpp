#include "include/Widget/solverpropertywidget.h"
#include "ui_solverpropertywidget.h"

SolverPropertyWidget::SolverPropertyWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SolverPropertyWidget)
{
    ui->setupUi(this);
}

SolverPropertyWidget::~SolverPropertyWidget()
{
    delete ui;
}
