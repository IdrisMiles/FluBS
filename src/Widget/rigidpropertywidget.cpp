#include "include/Widget/rigidpropertywidget.h"
#include "ui_rigidpropertywidget.h"

RigidPropertyWidget::RigidPropertyWidget(QWidget *parent, std::shared_ptr<RigidProperty> _property) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::RigidPropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
    SetProperty(_property);
}

RigidPropertyWidget::~RigidPropertyWidget()
{
    m_property = nullptr;
    delete ui;
}


void RigidPropertyWidget::SetProperty(std::shared_ptr<RigidProperty> _property)
{
    if(_property != nullptr)
    {
        SetNumParticles(_property->numParticles);
        SetParticleMass(_property->particleMass);
        SetParticleRadius(_property->particleRadius);
        SetRestDensity(_property->restDensity);

        ui->kinematic->setChecked(_property->kinematic);
        ui->static_2->setChecked(_property->m_static);

        m_property = _property;
    }
}

RigidProperty *RigidPropertyWidget::GetProperty()
{
    return m_property.get();
}

//-----------------------------------------------------------------------------------------------------------

void RigidPropertyWidget::OnPropertyChanged()
{
    if(m_property == nullptr)
    {
        m_property = std::shared_ptr<RigidProperty>(new RigidProperty());
    }

    if(m_property != nullptr)
    {
        m_property->kinematic = ui->kinematic->isChecked();
        m_property->m_static = ui->static_2->isChecked();

        m_property->numParticles = GetNumParticles();
        m_property->particleMass = GetParticleMass();
        m_property->particleRadius = GetParticleRadius();
        m_property->restDensity = GetRestDensity();

        emit PropertyChanged(m_property);
    }

}
