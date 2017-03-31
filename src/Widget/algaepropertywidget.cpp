#include "include/Widget/algaepropertywidget.h"
#include "ui_algaepropertywidget.h"

AlgaePropertyWidget::AlgaePropertyWidget(QWidget *parent) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::AlgaePropertyWidget)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
}

AlgaePropertyWidget::~AlgaePropertyWidget()
{
    delete ui;
}
