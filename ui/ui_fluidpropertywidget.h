/********************************************************************************
** Form generated from reading UI file 'fluidpropertywidget.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FLUIDPROPERTYWIDGET_H
#define UI_FLUIDPROPERTYWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FluidPropertyWidget
{
public:
    QWidget *layout;
    QGridLayout *gridLayout_2;
    QLabel *surfaceThresholdLabel;
    QLabel *viscosityLabel;
    QLabel *surfaceTensionLabel;
    QDoubleSpinBox *viscosity;
    QDoubleSpinBox *gasStiffness;
    QLabel *gasStiffnessLabel;
    QDoubleSpinBox *surfaceTension;
    QDoubleSpinBox *surfaceThreshold;
    QFrame *line;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *FluidPropertyWidget)
    {
        if (FluidPropertyWidget->objectName().isEmpty())
            FluidPropertyWidget->setObjectName(QStringLiteral("FluidPropertyWidget"));
        FluidPropertyWidget->resize(400, 300);
        layout = new QWidget(FluidPropertyWidget);
        layout->setObjectName(QStringLiteral("layout"));
        layout->setGeometry(QRect(9, 9, 199, 128));
        gridLayout_2 = new QGridLayout(layout);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        surfaceThresholdLabel = new QLabel(layout);
        surfaceThresholdLabel->setObjectName(QStringLiteral("surfaceThresholdLabel"));

        gridLayout_2->addWidget(surfaceThresholdLabel, 5, 0, 1, 1);

        viscosityLabel = new QLabel(layout);
        viscosityLabel->setObjectName(QStringLiteral("viscosityLabel"));

        gridLayout_2->addWidget(viscosityLabel, 3, 0, 1, 1);

        surfaceTensionLabel = new QLabel(layout);
        surfaceTensionLabel->setObjectName(QStringLiteral("surfaceTensionLabel"));

        gridLayout_2->addWidget(surfaceTensionLabel, 4, 0, 1, 1);

        viscosity = new QDoubleSpinBox(layout);
        viscosity->setObjectName(QStringLiteral("viscosity"));
        viscosity->setSingleStep(0.01);

        gridLayout_2->addWidget(viscosity, 3, 1, 1, 1);

        gasStiffness = new QDoubleSpinBox(layout);
        gasStiffness->setObjectName(QStringLiteral("gasStiffness"));
        gasStiffness->setMaximum(1000);

        gridLayout_2->addWidget(gasStiffness, 2, 1, 1, 1);

        gasStiffnessLabel = new QLabel(layout);
        gasStiffnessLabel->setObjectName(QStringLiteral("gasStiffnessLabel"));

        gridLayout_2->addWidget(gasStiffnessLabel, 2, 0, 1, 1);

        surfaceTension = new QDoubleSpinBox(layout);
        surfaceTension->setObjectName(QStringLiteral("surfaceTension"));
        surfaceTension->setSingleStep(0.01);

        gridLayout_2->addWidget(surfaceTension, 4, 1, 1, 1);

        surfaceThreshold = new QDoubleSpinBox(layout);
        surfaceThreshold->setObjectName(QStringLiteral("surfaceThreshold"));

        gridLayout_2->addWidget(surfaceThreshold, 5, 1, 1, 1);

        line = new QFrame(layout);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(line, 0, 0, 1, 2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 6, 0, 1, 1);


        retranslateUi(FluidPropertyWidget);

        QMetaObject::connectSlotsByName(FluidPropertyWidget);
    } // setupUi

    void retranslateUi(QWidget *FluidPropertyWidget)
    {
        FluidPropertyWidget->setWindowTitle(QApplication::translate("FluidPropertyWidget", "Form", 0));
        surfaceThresholdLabel->setText(QApplication::translate("FluidPropertyWidget", "Surface Threshold", 0));
        viscosityLabel->setText(QApplication::translate("FluidPropertyWidget", "Viscosity", 0));
        surfaceTensionLabel->setText(QApplication::translate("FluidPropertyWidget", "Surface Tension", 0));
        gasStiffnessLabel->setText(QApplication::translate("FluidPropertyWidget", "Gass Stiffness", 0));
    } // retranslateUi

};

namespace Ui {
    class FluidPropertyWidget: public Ui_FluidPropertyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FLUIDPROPERTYWIDGET_H
