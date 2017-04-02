#include "include/Widget/algaepropertywidget.h"
#include "ui_algaepropertywidget.h"

AlgaePropertyWidget::AlgaePropertyWidget(QWidget *parent) :
    SphParticlePropertyWidget(parent),
    ui(new Ui::AlgaePropertyWidget)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
}

AlgaePropertyWidget::~AlgaePropertyWidget()
{
    delete ui;
}

void AlgaePropertyWidget::SetProperty(std::shared_ptr<AlgaeProperty> _algaeProperty)
{
    m_property = _algaeProperty;
}

AlgaeProperty *AlgaePropertyWidget::GetProperty()
{
    return m_property.get();
}
