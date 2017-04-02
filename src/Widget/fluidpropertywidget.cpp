#include "include/Widget/fluidpropertywidget.h"
#include "ui_fluidpropertywidget.h"

FluidPropertyWidget::FluidPropertyWidget(QWidget *parent, std::shared_ptr<FluidProperty> _property) :
    SphParticlePropertyWidget(parent, _property),
    ui(new Ui::FluidPropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);

}

FluidPropertyWidget::~FluidPropertyWidget()
{
    delete ui;
}

void FluidPropertyWidget::SetProperty(std::shared_ptr<FluidProperty> _fluidProperty)
{
    m_property = _fluidProperty;
}

FluidProperty *FluidPropertyWidget::GetProperty()
{
    return m_property.get();
}
