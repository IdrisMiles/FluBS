#include "include/Widget/rigidpropertywidget.h"
#include "ui_rigidpropertywidget.h"

//-----------------------------------------------------------------------------------------------------------

RigidPropertyWidget::RigidPropertyWidget(QWidget *parent, RigidProperty _property, float posX, float posY, float posZ, float rotX, float rotY, float rotZ) :
    SphParticlePropertyWidget(parent, _property),
    ui(new Ui::RigidPropertyWidget),
    m_property(_property)
{
    ui->setupUi(this);

    AddWidgetToGridLayout(ui->layout, 0, 1, 2);
    SetProperty(_property);
    SetTransform(posX, posY, posZ, rotX, rotY, rotZ);

    connect(ui->static_2, &QCheckBox::clicked, this, &RigidPropertyWidget::OnPropertyChanged);
    connect(ui->kinematic, &QCheckBox::clicked, this, &RigidPropertyWidget::OnPropertyChanged);

    connect(ui->posX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &RigidPropertyWidget::OnTransformChanged);
    connect(ui->posY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &RigidPropertyWidget::OnTransformChanged);
    connect(ui->posZ, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &RigidPropertyWidget::OnTransformChanged);

    connect(ui->rotX, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &RigidPropertyWidget::OnTransformChanged);
    connect(ui->rotY, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &RigidPropertyWidget::OnTransformChanged);
    connect(ui->rotZ, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &RigidPropertyWidget::OnTransformChanged);
}

//-----------------------------------------------------------------------------------------------------------

RigidPropertyWidget::~RigidPropertyWidget()
{

    delete ui;
}

//-----------------------------------------------------------------------------------------------------------

void RigidPropertyWidget::SetProperty(RigidProperty _property)
{

    m_property = _property;
    SphParticlePropertyWidget::SetProperty(_property);

    SetNumParticles(_property.numParticles);
    SetParticleMass(_property.particleMass);
    SetParticleRadius(_property.particleRadius);
    SetRestDensity(_property.restDensity);

    ui->kinematic->setChecked(_property.kinematic);
    ui->static_2->setChecked(_property.m_static);
}

//-----------------------------------------------------------------------------------------------------------

RigidProperty RigidPropertyWidget::GetProperty()
{
    return m_property;
}

//---------------------------------------------------------------------------------------

void RigidPropertyWidget::SetTransform(float posX, float posY, float posZ, float rotX, float rotY, float rotZ)
{
    ui->posX->setValue(posX);
    ui->posY->setValue(posY);
    ui->posZ->setValue(posZ);

    ui->rotX->setValue(rotX);
    ui->rotY->setValue(rotY);
    ui->rotZ->setValue(rotZ);
}

//-----------------------------------------------------------------------------------------------------------

void RigidPropertyWidget::OnPropertyChanged()
{
        m_property.kinematic = ui->kinematic->isChecked();
        m_property.m_static = ui->static_2->isChecked();

        m_property.numParticles = GetNumParticles();
        m_property.particleMass = GetParticleMass();
        m_property.particleRadius = GetParticleRadius();
        m_property.restDensity = GetRestDensity();


        float dia = 2.0f * m_property.particleRadius;
        m_property.particleMass = m_property.restDensity * (dia * dia * dia);
        SetParticleMass(m_property.particleMass);

        emit PropertyChanged(m_property);

}

//-----------------------------------------------------------------------------------------------------------

void RigidPropertyWidget::OnTransformChanged()
{
    emit TransformChanged(ui->posX->value(), ui->posY->value(), ui->posZ->value(),
                          ui->rotX->value(), ui->rotY->value(), ui->rotZ->value(),
                          ui->scaleX->value(), ui->scaleY->value(), ui->scaleZ->value());

    emit PropertyChanged(m_property);
}

//-----------------------------------------------------------------------------------------------------------
