#include "include/Widget/sphparticlepropertywidget.h"
#include "ui_sphparticlepropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

SphParticlePropertyWidget::SphParticlePropertyWidget(QWidget *parent, SphParticleProperty *_property) :
    QWidget(parent),
    ui(new Ui::SphParticlePropertyWidget),
    m_numRow(5),
    m_property(_property)
{
    ui->setupUi(this);
    SetProperty(_property);

    connect(ui->numParticles, QOverload<int>::of(&QSpinBox::valueChanged), this, &SphParticlePropertyWidget::OnPropertyChanged);
    connect(ui->particleMass, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SphParticlePropertyWidget::OnPropertyChanged);
    connect(ui->particleRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SphParticlePropertyWidget::OnPropertyChanged);
    connect(ui->restDensity, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SphParticlePropertyWidget::OnPropertyChanged);
    connect(ui->smoothingLength, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SphParticlePropertyWidget::OnPropertyChanged);
}

//-----------------------------------------------------------------------------------------------------------

SphParticlePropertyWidget::~SphParticlePropertyWidget()
{
    m_property = nullptr;
    delete ui;
}

//-----------------------------------------------------------------------------------------------------------


void SphParticlePropertyWidget::AddWidgetToGridLayout(QWidget *w, int col, int rowSpan, int colSpan)
{
    ui->gridLayout->addWidget(w, m_numRow++, col, rowSpan, colSpan);
}

//-----------------------------------------------------------------------------------------------------------

void SphParticlePropertyWidget::SetProperty(SphParticleProperty *_property)
{
    if(_property != nullptr)
    {
        m_property = _property;
        ui->numParticles->setValue((int)m_property->numParticles);
        ui->particleMass->setValue(m_property->particleMass);
        ui->particleRadius->setValue(m_property->particleRadius);
        ui->restDensity->setValue(m_property->restDensity);
    }
}

//-----------------------------------------------------------------------------------------------------------

SphParticleProperty *SphParticlePropertyWidget::GetProperty()
{
    return m_property;
}

//-----------------------------------------------------------------------------------------------------------


void SphParticlePropertyWidget::OnPropertyChanged()
{
    if(m_property == nullptr)
    {
        m_property = new SphParticleProperty();
    }

    if(m_property != nullptr)
    {

        m_property->numParticles = GetNumParticles();
        m_property->particleMass = GetParticleMass();
        m_property->particleRadius = GetParticleRadius();
        m_property->restDensity = GetRestDensity();
    }
    emit PropertyChanged(m_property);
}

//-----------------------------------------------------------------------------------------------------------


void SphParticlePropertyWidget::SetNumParticles(const int _numParticles)
{
    ui->numParticles->setValue(_numParticles);
}

//-----------------------------------------------------------------------------------------------------------

void SphParticlePropertyWidget::SetParticleMass(const float _particlesMass)
{
    ui->particleMass->setValue(_particlesMass);
}

//-----------------------------------------------------------------------------------------------------------

void SphParticlePropertyWidget::SetParticleRadius(const float _particlesRadius)
{
    ui->particleRadius->setValue(_particlesRadius);
}

//-----------------------------------------------------------------------------------------------------------

void SphParticlePropertyWidget::SetRestDensity(const float _restDensity)
{
    ui->restDensity->setValue(_restDensity);
}

//-----------------------------------------------------------------------------------------------------------

int SphParticlePropertyWidget::GetNumParticles()
{
    return ui->numParticles->value();
}

//-----------------------------------------------------------------------------------------------------------

float SphParticlePropertyWidget::GetParticleMass()
{
    return ui->particleMass->value();
}

//-----------------------------------------------------------------------------------------------------------

float SphParticlePropertyWidget::GetParticleRadius()
{
    return ui->particleRadius->value();
}

//-----------------------------------------------------------------------------------------------------------

float SphParticlePropertyWidget::GetRestDensity()
{
    return ui->restDensity->value();
}

//-----------------------------------------------------------------------------------------------------------
