#include "include/fluidpropertywidget.h"

FluidPropertyWidget::FluidPropertyWidget(QWidget *parent) : QWidget(parent)
{
    m_gridLayout = std::shared_ptr<QGridLayout>(new QGridLayout(this));

    surfaceTensionLabel = std::shared_ptr<QLabel>(new QLabel("Surface Tension", this));
    surfaceTension = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    surfaceThresholdLabel = std::shared_ptr<QLabel>(new QLabel("Surface Threshold", this));
    surfaceThreshold = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    gasStiffnessLabel = std::shared_ptr<QLabel>(new QLabel("Gass Stiffness", this));
    gasStiffness = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));
    viscosityLabel = std::shared_ptr<QLabel>(new QLabel("Viscosity", this));
    viscosity = std::shared_ptr<QDoubleSpinBox>(new QDoubleSpinBox(this));

    int row = 0;
    m_gridLayout->addWidget(surfaceTensionLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(surfaceTension.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(surfaceThresholdLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(surfaceThreshold.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(gasStiffnessLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(gasStiffness.get(), row++, 1, 1, 1);

    m_gridLayout->addWidget(viscosityLabel.get(), row, 0, 1, 1);
    m_gridLayout->addWidget(viscosity.get(), row++, 1, 1, 1);

    this->setLayout(m_gridLayout.get());
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
    m_fluidProperty = _fluidProperty;
}

std::shared_ptr<FluidProperty> FluidPropertyWidget::GetFluidProperty()
{
    return m_fluidProperty;
}
