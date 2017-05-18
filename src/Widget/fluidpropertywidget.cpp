#include "include/Widget/fluidpropertywidget.h"
#include "ui_fluidpropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

FluidPropertyWidget::FluidPropertyWidget(QWidget *parent, FluidProperty _property) :
    SphParticlePropertyWidget(parent, _property),
    ui(new Ui::FluidPropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);

    SetProperty(_property);



    connect(ui->gasStiffness, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &FluidPropertyWidget::OnPropertyChanged);
    connect(ui->viscosity, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &FluidPropertyWidget::OnPropertyChanged);
    connect(ui->surfaceTension, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &FluidPropertyWidget::OnPropertyChanged);
    connect(ui->surfaceThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &FluidPropertyWidget::OnPropertyChanged);


}

//-----------------------------------------------------------------------------------------------------------

FluidPropertyWidget::~FluidPropertyWidget()
{
    delete ui;
}

//-----------------------------------------------------------------------------------------------------------

void FluidPropertyWidget::SetProperty(FluidProperty _property)
{

    m_property = _property;
    SphParticlePropertyWidget::SetProperty(_property);

    ui->surfaceTension->setValue((double)_property.surfaceTension);
    ui->surfaceThreshold->setValue((double)_property.surfaceThreshold);
    ui->viscosity->setValue((double)_property.viscosity);
    ui->gasStiffness->setValue((double)_property.gasStiffness);

}

//-----------------------------------------------------------------------------------------------------------

FluidProperty FluidPropertyWidget::GetProperty()
{
    return m_property;
}

//-----------------------------------------------------------------------------------------------------------

void FluidPropertyWidget::OnPropertyChanged()
{
        m_property.numParticles = GetNumParticles();
        m_property.particleRadius = GetParticleRadius();
        m_property.restDensity = GetRestDensity();

        m_property.surfaceTension = ui->surfaceTension->value();
        m_property.surfaceThreshold = ui->surfaceThreshold->value();
        m_property.viscosity = ui->viscosity->value();
        m_property.gasStiffness = ui->gasStiffness->value();


        float dia = 2.0f * m_property.particleRadius;
        m_property.particleMass = m_property.restDensity * (dia * dia * dia);
        SetParticleMass(m_property.particleMass);


    emit PropertyChanged(m_property);
}
