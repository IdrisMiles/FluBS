/********************************************************************************
** Form generated from reading UI file 'algaepropertywidget.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ALGAEPROPERTYWIDGET_H
#define UI_ALGAEPROPERTYWIDGET_H

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

class Ui_AlgaePropertyWidget
{
public:
    QWidget *layout;
    QGridLayout *gridLayout;
    QDoubleSpinBox *bioIntensity;
    QFrame *line;
    QLabel *bioIntensityLabel;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *AlgaePropertyWidget)
    {
        if (AlgaePropertyWidget->objectName().isEmpty())
            AlgaePropertyWidget->setObjectName(QStringLiteral("AlgaePropertyWidget"));
        AlgaePropertyWidget->resize(400, 300);
        layout = new QWidget(AlgaePropertyWidget);
        layout->setObjectName(QStringLiteral("layout"));
        layout->setGeometry(QRect(40, 80, 271, 51));
        gridLayout = new QGridLayout(layout);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        bioIntensity = new QDoubleSpinBox(layout);
        bioIntensity->setObjectName(QStringLiteral("bioIntensity"));

        gridLayout->addWidget(bioIntensity, 1, 1, 1, 1);

        line = new QFrame(layout);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line, 0, 0, 1, 2);

        bioIntensityLabel = new QLabel(layout);
        bioIntensityLabel->setObjectName(QStringLiteral("bioIntensityLabel"));

        gridLayout->addWidget(bioIntensityLabel, 1, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 2, 0, 1, 1);


        retranslateUi(AlgaePropertyWidget);

        QMetaObject::connectSlotsByName(AlgaePropertyWidget);
    } // setupUi

    void retranslateUi(QWidget *AlgaePropertyWidget)
    {
        AlgaePropertyWidget->setWindowTitle(QApplication::translate("AlgaePropertyWidget", "Form", 0));
        bioIntensityLabel->setText(QApplication::translate("AlgaePropertyWidget", "Bioluminescence Intensity", 0));
    } // retranslateUi

};

namespace Ui {
    class AlgaePropertyWidget: public Ui_AlgaePropertyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ALGAEPROPERTYWIDGET_H
