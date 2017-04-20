#include "include/Widget/rigidpropertywidget.h"
#include "ui_rigidpropertywidget.h"

RigidPropertyWidget::RigidPropertyWidget(QWidget *parent, std::shared_ptr<RigidProperty> _property) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::RigidPropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
}

RigidPropertyWidget::~RigidPropertyWidget()
{
    m_property = nullptr;
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

//-----------------------------------------------------------------------------------------------------------

void RigidPropertyWidget::OnPropertyChanged()
{
    if(m_property != nullptr)
    {

        emit PropertyChanged(m_property);
    }

}
