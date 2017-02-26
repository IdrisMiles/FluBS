#ifndef FLUIDPROPERTYWIDGET_H
#define FLUIDPROPERTYWIDGET_H

#include <QGroupBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QGridLayout>
#include <QPushButton>

#include <memory>
#include "SPH/fluidproperty.h"

class FluidPropertyWidget : public QWidget
{
    Q_OBJECT

public:
    FluidPropertyWidget(QWidget *parent = nullptr);
    ~FluidPropertyWidget();

    void SetFluidProperty(std::shared_ptr<FluidProperty> _fluidProperty);
    std::shared_ptr<FluidProperty> GetFluidProperty();

signals:
    void ResetSim();

public slots:
    void ResetSimBtnClicked();
    void TogglePlaySimBtnClicked();
    void UpdateFluidProperty();



private:
    std::shared_ptr<FluidProperty> m_fluidProperty;

    std::shared_ptr<QGridLayout> m_gridLayout;

    std::shared_ptr<QDoubleSpinBox> numParticles;
    std::shared_ptr<QDoubleSpinBox> smoothingLength;
    std::shared_ptr<QDoubleSpinBox> particleRadius;
    std::shared_ptr<QDoubleSpinBox> particleMass;
    std::shared_ptr<QDoubleSpinBox> restDensity;

    std::shared_ptr<QLabel> surfaceTensionLabel;
    std::shared_ptr<QDoubleSpinBox> surfaceTension;
    std::shared_ptr<QLabel> surfaceThresholdLabel;
    std::shared_ptr<QDoubleSpinBox> surfaceThreshold;
    std::shared_ptr<QLabel> gasStiffnessLabel;
    std::shared_ptr<QDoubleSpinBox> gasStiffness;
    std::shared_ptr<QLabel> viscosityLabel;
    std::shared_ptr<QDoubleSpinBox> viscosity;

    std::shared_ptr<QLabel> deltaTimeLabel;
    std::shared_ptr<QDoubleSpinBox> deltaTime;
    std::shared_ptr<QDoubleSpinBox> solveIterations;
    std::shared_ptr<QDoubleSpinBox> gridResolution;
    std::shared_ptr<QDoubleSpinBox> gridCellWidth;

    std::shared_ptr<QPushButton> reset;
    std::shared_ptr<QPushButton> togglePlay;
};

#endif // FLUIDPROPERTYWIDGET_H
