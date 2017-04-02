#include "include/Widget/sphparticlepropertywidget.h"
#include "ui_sphparticlepropertywidget.h"

SphParticlePropertyWidget::SphParticlePropertyWidget(QWidget *parent, std::shared_ptr<SphParticleProperty> _property) :
    QWidget(parent),
    ui(new Ui::SphParticlePropertyWidget),
    m_numRow(5),
    m_property(_property)
{
    ui->setupUi(this);
}

SphParticlePropertyWidget::~SphParticlePropertyWidget()
{
    delete ui;
}


void SphParticlePropertyWidget::AddWidgetToGridLayout(QWidget *w, int col, int rowSpan, int colSpan)
{
    ui->gridLayout->addWidget(w, m_numRow++, col, rowSpan, colSpan);
}

void SphParticlePropertyWidget::SetProperty(std::shared_ptr<SphParticleProperty> _property)
{
    m_property = _property;
}

SphParticleProperty *SphParticlePropertyWidget::GetProperty()
{
    return m_property.get();
}
