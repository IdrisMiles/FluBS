#include "include/Widget/solverpropertywidget.h"
#include "ui_solverpropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

SolverPropertyWidget::SolverPropertyWidget(QWidget *parent, std::shared_ptr<FluidSolverProperty> _property) :
    QWidget(parent),
    ui(new Ui::SolverPropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);
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
    m_property = _property;
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

        emit PropertyChanged(m_property);
    }

}
