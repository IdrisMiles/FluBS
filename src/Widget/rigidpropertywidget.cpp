#include "include/Widget/rigidpropertywidget.h"
#include "ui_rigidpropertywidget.h"

RigidPropertyWidget::RigidPropertyWidget(QWidget *parent) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::RigidPropertyWidget)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
}

RigidPropertyWidget::~RigidPropertyWidget()
{
    delete ui;
}


void RigidPropertyWidget::SetProperty(std::shared_ptr<RigidProperty> _rigidProperty)
{
    m_property = _rigidProperty;
}

RigidProperty *RigidPropertyWidget::GetProperty()
{
    return m_property.get();
}
