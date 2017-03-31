#include "include/Widget/fluidpropertywidget.h"
#include "ui_fluidpropertywidget.h"

FluidPropertyWidget::FluidPropertyWidget(QWidget *parent) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::FluidPropertyWidget)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);

}

FluidPropertyWidget::~FluidPropertyWidget()
{
    delete ui;
}
