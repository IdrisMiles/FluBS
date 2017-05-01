#include "include/Widget/algaepropertywidget.h"
#include "ui_algaepropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

AlgaePropertyWidget::AlgaePropertyWidget(QWidget *parent, std::shared_ptr<AlgaeProperty> _property) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::AlgaePropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
    SetProperty(_property);
}

//-----------------------------------------------------------------------------------------------------------

AlgaePropertyWidget::~AlgaePropertyWidget()
{
    m_property = nullptr;
    delete ui;
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::SetProperty(std::shared_ptr<AlgaeProperty> _property)
{
    if(_property != nullptr)
    {
        SetNumParticles(_property->numParticles);
        SetParticleMass(_property->particleMass);
        SetParticleRadius(_property->particleRadius);
        SetRestDensity(_property->restDensity);


        m_property = _property;
    }
}

//-----------------------------------------------------------------------------------------------------------

AlgaeProperty *AlgaePropertyWidget::GetProperty()
{
    return m_property.get();
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::OnPropertyChanged()
{
    if(m_property == nullptr)
    {
        m_property = std::shared_ptr<AlgaeProperty>(new AlgaeProperty());
    }

    if(m_property != nullptr)
    {

        m_property->numParticles = GetNumParticles();
        m_property->particleMass = GetParticleMass();
        m_property->particleRadius = GetParticleRadius();
        m_property->restDensity = GetRestDensity();

        emit PropertyChanged(m_property);
    }

}
