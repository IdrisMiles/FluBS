#include "include/Widget/algaepropertywidget.h"
#include "ui_algaepropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

AlgaePropertyWidget::AlgaePropertyWidget(QWidget *parent, AlgaeProperty _property) :
    SphParticlePropertyWidget(parent, _property),
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
    delete ui;
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::SetProperty(AlgaeProperty _property)
{

    m_property = _property;
    SphParticlePropertyWidget::SetProperty(_property);

    ui->bioThreshold->setValue(m_property.bioluminescenceThreshold);
    ui->reactionRate->setValue(m_property.reactionRate);
    ui->deactionRate->setValue(m_property.deactionRate);
}

//-----------------------------------------------------------------------------------------------------------

AlgaeProperty AlgaePropertyWidget::GetProperty()
{
    return m_property;
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::OnPropertyChanged()
{

        m_property.numParticles = GetNumParticles();
        m_property.particleRadius = GetParticleRadius();
        m_property.restDensity = GetRestDensity();
        m_property.bioluminescenceThreshold = ui->bioThreshold->value();
        m_property.reactionRate = ui->reactionRate->value();
        m_property.deactionRate = ui->deactionRate->value();


        float dia = 2.0f * m_property.particleRadius;
        m_property.particleMass = m_property.restDensity * (dia * dia * dia);
        SetParticleMass(m_property.particleMass);


    emit PropertyChanged(m_property);
}
