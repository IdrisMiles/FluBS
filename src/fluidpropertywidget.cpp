#include "include/fluidpropertywidget.h"
#include <float.h>
FluidPropertyWidget::FluidPropertyWidget(QWidget *parent) : QWidget(parent)
{
    m_gridLayout = std::shared_ptr<QGridLayout>(new QGridLayout(this));

    reset = std::shared_ptr<QPushButton>(new QPushButton("Reset", this));
    togglePlay = std::shared_ptr<QPushButton>(new QPushButton("Pause", this));

    surfaceTensionLabel = std::shared_ptr<QLabel>(new QLabel("Surface Tension", this));
    surfaceTension = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    surfaceTension->setMinimum(FLT_MIN);
    surfaceTension->setMaximum(FLT_MAX);
    surfaceTension->setDecimals(10);

    surfaceThresholdLabel = std::shared_ptr<QLabel>(new QLabel("Surface Threshold", this));
    surfaceThreshold = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    surfaceThreshold->setMinimum(FLT_MIN);
    surfaceThreshold->setMaximum(FLT_MAX);
    surfaceThreshold->setDecimals(10);

    gasStiffnessLabel = std::shared_ptr<QLabel>(new QLabel("Gass Stiffness", this));
    gasStiffness = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    gasStiffness->setMinimum(FLT_MIN);
    gasStiffness->setMaximum(FLT_MAX);
    gasStiffness->setDecimals(10);

    viscosityLabel = std::shared_ptr<QLabel>(new QLabel("Viscosity", this));
    viscosity = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    viscosity->setMinimum(FLT_MIN);
    viscosity->setMaximum(FLT_MAX);
    viscosity->setDecimals(10);

    deltaTimeLabel = std::shared_ptr<QLabel>(new QLabel("Delta Time", this));
    deltaTime = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    deltaTime->setMinimum(FLT_MIN);
    deltaTime->setMaximum(1.0);
    deltaTime->setDecimals(10);

    int row = 0;
    m_gridLayout->addWidget(surfaceTensionLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(surfaceTension.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(surfaceThresholdLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(surfaceThreshold.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(gasStiffnessLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(gasStiffness.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(viscosityLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(viscosity.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(deltaTimeLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(deltaTime.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(reset.get(), row++, 1, 1, 1);
    m_gridLayout->addWidget(togglePlay.get(), row++, 1, 1, 1);

    this->setLayout(m_gridLayout.get());


    connect(surfaceTension.get(), SIGNAL(valueChanged(double)), this, SLOT(UpdateFluidProperty()));
    connect(surfaceThreshold.get(), SIGNAL(valueChanged(double)), this, SLOT(UpdateFluidProperty()));
    connect(viscosity.get(), SIGNAL(valueChanged(double)), this, SLOT(UpdateFluidProperty()));
    connect(gasStiffness.get(), SIGNAL(valueChanged(double)), this, SLOT(UpdateFluidProperty()));
    connect(deltaTime.get(), SIGNAL(valueChanged(double)), this, SLOT(UpdateFluidProperty()));


    connect(reset.get(), SIGNAL(clicked(bool)), this, SLOT(ResetSimBtnClicked()));
    connect(togglePlay.get(), SIGNAL(clicked(bool)), this, SLOT(TogglePlaySimBtnClicked()));
}

FluidPropertyWidget::~FluidPropertyWidget()
{
    m_fluidProperty = nullptr;

    m_gridLayout = nullptr;

    numParticles = nullptr;
    smoothingLength = nullptr;
    particleRadius = nullptr;
    particleMass = nullptr;
    restDensity = nullptr;
    surfaceTensionLabel = nullptr;
    surfaceTension = nullptr;
    surfaceThresholdLabel = nullptr;
    surfaceThreshold = nullptr;
    gasStiffnessLabel = nullptr;
    gasStiffness = nullptr;
    viscosityLabel = nullptr;
    viscosity = nullptr;
    deltaTime = nullptr;
    solveIterations = nullptr;
    gridResolution = nullptr;
    gridCellWidth = nullptr;
}


void FluidPropertyWidget::SetFluidProperty(std::shared_ptr<FluidProperty> _fluidProperty)
{
    if(_fluidProperty != nullptr)
    {
        m_fluidProperty = _fluidProperty;

        surfaceTension->setValue((double)m_fluidProperty->surfaceTension);
        surfaceThreshold->setValue((double)m_fluidProperty->surfaceThreshold);
        viscosity->setValue((double)m_fluidProperty->viscosity);
        gasStiffness->setValue((double)m_fluidProperty->gasStiffness);

        deltaTime->setValue((double)m_fluidProperty->deltaTime);

    }
}

std::shared_ptr<FluidProperty> FluidPropertyWidget::GetFluidProperty()
{
    return m_fluidProperty;
}


void FluidPropertyWidget::UpdateFluidProperty()
{
    if(m_fluidProperty != nullptr)
    {
        m_fluidProperty->surfaceTension = surfaceTension->value();
        m_fluidProperty->surfaceThreshold = surfaceThreshold->value();
        m_fluidProperty->viscosity = viscosity->value();
        m_fluidProperty->gasStiffness = gasStiffness->value();

        m_fluidProperty->deltaTime = deltaTime->value();

        float dia = 2.0f * m_fluidProperty->particleRadius;
        m_fluidProperty->particleMass = m_fluidProperty->restDensity * (dia * dia * dia);
    }
}


void FluidPropertyWidget::ResetSimBtnClicked()
{
    emit ResetSim();
}

void FluidPropertyWidget::TogglePlaySimBtnClicked()
{
    m_fluidProperty->play = !m_fluidProperty->play;
    if(m_fluidProperty->play)
    {
        togglePlay->setText("Pause");
    }
    else
    {
        togglePlay->setText("Play");
    }
}
