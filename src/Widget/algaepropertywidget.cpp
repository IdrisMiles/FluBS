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
}

//-----------------------------------------------------------------------------------------------------------

AlgaePropertyWidget::~AlgaePropertyWidget()
{
    m_property = nullptr;
    delete ui;
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::SetProperty(std::shared_ptr<AlgaeProperty> _algaeProperty)
{
    m_property = _algaeProperty;
}

//-----------------------------------------------------------------------------------------------------------

AlgaeProperty *AlgaePropertyWidget::GetProperty()
{
    return m_property.get();
}

//-----------------------------------------------------------------------------------------------------------

void AlgaePropertyWidget::OnPropertyChanged()
{
    if(m_property != nullptr)
    {

        emit PropertyChanged(m_property);
    }

}
