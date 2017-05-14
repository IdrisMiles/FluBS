#include "include/Widget/algaepropertywidget.h"
#include "ui_algaepropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

AlgaePropertyWidget::AlgaePropertyWidget(QWidget *parent, AlgaeProperty *_property) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::AlgaePropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
    SetProperty(_property);

    connect(ui->bioThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &AlgaePropertyWidget::OnPropertyChanged);
    connect(ui->reactionRate, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &AlgaePropertyWidget::OnPropertyChanged);
    connect(ui->deactionRate, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &AlgaePropertyWidget::OnPropertyChanged);
}

//-----------------------------------------------------------------------------------------------------------

AlgaePropertyWidget::~AlgaePropertyWidget()
{
    m_property = nullptr;
    delete ui;
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::SetProperty(AlgaeProperty *_property)
{
    if(_property != nullptr)
    {
        SetNumParticles(_property->numParticles);
        SetParticleMass(_property->particleMass);
        SetParticleRadius(_property->particleRadius);
        SetRestDensity(_property->restDensity);

        ui->bioThreshold->setValue(_property->bioluminescenceThreshold);
        ui->reactionRate->setValue(_property->reactionRate);
        ui->deactionRate->setValue(_property->deactionRate);


        m_property = _property;
    }
}

//-----------------------------------------------------------------------------------------------------------

AlgaeProperty *AlgaePropertyWidget::GetProperty()
{
    return m_property;
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::OnPropertyChanged()
{
    if(m_property == nullptr)
    {
        m_property = new AlgaeProperty();
    }

    if(m_property != nullptr)
    {

        m_property->numParticles = GetNumParticles();
        m_property->particleMass = GetParticleMass();
        m_property->particleRadius = GetParticleRadius();
        m_property->restDensity = GetRestDensity();
        m_property->bioluminescenceThreshold = ui->bioThreshold->value();
        m_property->reactionRate = ui->reactionRate->value();
        m_property->deactionRate = ui->deactionRate->value();

        emit PropertyChanged(m_property);
    }

}
