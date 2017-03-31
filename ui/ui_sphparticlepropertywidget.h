/********************************************************************************
** Form generated from reading UI file 'sphparticlepropertywidget.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SPHPARTICLEPROPERTYWIDGET_H
#define UI_SPHPARTICLEPROPERTYWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SphParticlePropertyWidget
{
public:
    QGridLayout *gridLayout;
    QLabel *particleMassLabel;
    QLabel *numParticlesLabel;
    QSpinBox *numParticles;
    QDoubleSpinBox *smoothingLength;
    QDoubleSpinBox *particleRadius;
    QDoubleSpinBox *restDensity;
    QDoubleSpinBox *particleMass;
    QLabel *smoothingLengthLabel;
    QLabel *particleRadiusLabel;
    QLabel *restDensityLabel;

    void setupUi(QWidget *SphParticlePropertyWidget)
    {
        if (SphParticlePropertyWidget->objectName().isEmpty())
            SphParticlePropertyWidget->setObjectName(QStringLiteral("SphParticlePropertyWidget"));
        SphParticlePropertyWidget->resize(400, 300);
        gridLayout = new QGridLayout(SphParticlePropertyWidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        particleMassLabel = new QLabel(SphParticlePropertyWidget);
        particleMassLabel->setObjectName(QStringLiteral("particleMassLabel"));

        gridLayout->addWidget(particleMassLabel, 1, 0, 1, 1);

        numParticlesLabel = new QLabel(SphParticlePropertyWidget);
        numParticlesLabel->setObjectName(QStringLiteral("numParticlesLabel"));

        gridLayout->addWidget(numParticlesLabel, 0, 0, 1, 1);

        numParticles = new QSpinBox(SphParticlePropertyWidget);
        numParticles->setObjectName(QStringLiteral("numParticles"));

        gridLayout->addWidget(numParticles, 0, 1, 1, 1);

        smoothingLength = new QDoubleSpinBox(SphParticlePropertyWidget);
        smoothingLength->setObjectName(QStringLiteral("smoothingLength"));

        gridLayout->addWidget(smoothingLength, 4, 1, 1, 1);

        particleRadius = new QDoubleSpinBox(SphParticlePropertyWidget);
        particleRadius->setObjectName(QStringLiteral("particleRadius"));

        gridLayout->addWidget(particleRadius, 2, 1, 1, 1);

        restDensity = new QDoubleSpinBox(SphParticlePropertyWidget);
        restDensity->setObjectName(QStringLiteral("restDensity"));

        gridLayout->addWidget(restDensity, 3, 1, 1, 1);

        particleMass = new QDoubleSpinBox(SphParticlePropertyWidget);
        particleMass->setObjectName(QStringLiteral("particleMass"));

        gridLayout->addWidget(particleMass, 1, 1, 1, 1);

        smoothingLengthLabel = new QLabel(SphParticlePropertyWidget);
        smoothingLengthLabel->setObjectName(QStringLiteral("smoothingLengthLabel"));

        gridLayout->addWidget(smoothingLengthLabel, 4, 0, 1, 1);

        particleRadiusLabel = new QLabel(SphParticlePropertyWidget);
        particleRadiusLabel->setObjectName(QStringLiteral("particleRadiusLabel"));

        gridLayout->addWidget(particleRadiusLabel, 2, 0, 1, 1);

        restDensityLabel = new QLabel(SphParticlePropertyWidget);
        restDensityLabel->setObjectName(QStringLiteral("restDensityLabel"));

        gridLayout->addWidget(restDensityLabel, 3, 0, 1, 1);


        retranslateUi(SphParticlePropertyWidget);

        QMetaObject::connectSlotsByName(SphParticlePropertyWidget);
    } // setupUi

    void retranslateUi(QWidget *SphParticlePropertyWidget)
    {
        SphParticlePropertyWidget->setWindowTitle(QApplication::translate("SphParticlePropertyWidget", "Form", 0));
        particleMassLabel->setText(QApplication::translate("SphParticlePropertyWidget", "Particle Mass", 0));
        numParticlesLabel->setText(QApplication::translate("SphParticlePropertyWidget", "Number of Particles", 0));
        smoothingLengthLabel->setText(QApplication::translate("SphParticlePropertyWidget", "Smoothing Length", 0));
        particleRadiusLabel->setText(QApplication::translate("SphParticlePropertyWidget", "Particle Radius", 0));
        restDensityLabel->setText(QApplication::translate("SphParticlePropertyWidget", "Rest Density", 0));
    } // retranslateUi

};

namespace Ui {
    class SphParticlePropertyWidget: public Ui_SphParticlePropertyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SPHPARTICLEPROPERTYWIDGET_H
