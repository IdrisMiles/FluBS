#include "include/Widget/solverpropertywidget.h"
#include "ui_solverpropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

SolverPropertyWidget::SolverPropertyWidget(QWidget *parent, std::shared_ptr<FluidSolverProperty> _property) :
    QWidget(parent),
    ui(new Ui::SolverPropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    SetProperty(_property);


    connect(ui->deltaTime, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SolverPropertyWidget::OnPropertyChanged);
    connect(ui->gridCellSize, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SolverPropertyWidget::OnPropertyChanged);
    connect(ui->gridRes, QOverload<int>::of(&QSpinBox::valueChanged), this, &SolverPropertyWidget::OnPropertyChanged);
    connect(ui->smoothingLength, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &SolverPropertyWidget::OnPropertyChanged);
    connect(ui->solveIterations, QOverload<int>::of(&QSpinBox::valueChanged), this, &SolverPropertyWidget::OnPropertyChanged);

}

//-----------------------------------------------------------------------------------------------------------

SolverPropertyWidget::~SolverPropertyWidget()
{
    m_property = nullptr;
    delete ui;
}

//-----------------------------------------------------------------------------------------------------------

void SolverPropertyWidget::SetProperty(std::shared_ptr<FluidSolverProperty> _property)
{
    if(_property != nullptr)
    {
        ui->deltaTime->setValue(_property->deltaTime);
        ui->gridCellSize->setValue(_property->gridCellWidth);
        ui->gridRes->setValue(_property->gridResolution);
        ui->smoothingLength->setValue(_property->smoothingLength);
        ui->solveIterations->setValue(_property->solveIterations);

        m_property = _property;
    }
}

//-----------------------------------------------------------------------------------------------------------

FluidSolverProperty *SolverPropertyWidget::GetProperty()
{
    return m_property.get();
}


//-----------------------------------------------------------------------------------------------------------

void SolverPropertyWidget::OnPropertyChanged()
{
    if(m_property != nullptr)
    {
        m_property->deltaTime = ui->deltaTime->value();
        m_property->gridCellWidth = ui->gridCellSize->value();
        m_property->gridResolution = ui->gridRes->value();
        m_property->smoothingLength = ui->smoothingLength->value();
        m_property->solveIterations = ui->solveIterations->value();

        emit PropertyChanged(m_property);
    }

}
