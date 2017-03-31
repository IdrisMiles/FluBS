#include "include/Widget/sphparticlepropertywidget.h"
#include "ui_sphparticlepropertywidget.h"

SphParticlePropertyWidget::SphParticlePropertyWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SphParticlePropertyWidget),
    m_numRow(5)
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
