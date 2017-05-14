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
    QLabel *reactionRateLabel;
    QDoubleSpinBox *bioThreshold;
    QFrame *line;
    QLabel *bioThresholdLabel;
    QSpacerItem *verticalSpacer;
    QLabel *deactionRateLabel;
    QDoubleSpinBox *reactionRate;
    QDoubleSpinBox *deactionRate;

    void setupUi(QWidget *AlgaePropertyWidget)
    {
        if (AlgaePropertyWidget->objectName().isEmpty())
            AlgaePropertyWidget->setObjectName(QStringLiteral("AlgaePropertyWidget"));
        AlgaePropertyWidget->resize(400, 300);
        layout = new QWidget(AlgaePropertyWidget);
        layout->setObjectName(QStringLiteral("layout"));
        layout->setGeometry(QRect(40, 80, 271, 161));
        gridLayout = new QGridLayout(layout);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        reactionRateLabel = new QLabel(layout);
        reactionRateLabel->setObjectName(QStringLiteral("reactionRateLabel"));

        gridLayout->addWidget(reactionRateLabel, 2, 0, 1, 1);

        bioThreshold = new QDoubleSpinBox(layout);
        bioThreshold->setObjectName(QStringLiteral("bioThreshold"));
        bioThreshold->setMaximum(1000);
        bioThreshold->setValue(200);

        gridLayout->addWidget(bioThreshold, 1, 1, 1, 1);

        line = new QFrame(layout);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line, 0, 0, 1, 2);

        bioThresholdLabel = new QLabel(layout);
        bioThresholdLabel->setObjectName(QStringLiteral("bioThresholdLabel"));

        gridLayout->addWidget(bioThresholdLabel, 1, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 4, 0, 1, 1);

        deactionRateLabel = new QLabel(layout);
        deactionRateLabel->setObjectName(QStringLiteral("deactionRateLabel"));

        gridLayout->addWidget(deactionRateLabel, 3, 0, 1, 1);

        reactionRate = new QDoubleSpinBox(layout);
        reactionRate->setObjectName(QStringLiteral("reactionRate"));
        reactionRate->setDecimals(5);
        reactionRate->setValue(0.1);

        gridLayout->addWidget(reactionRate, 2, 1, 1, 1);

        deactionRate = new QDoubleSpinBox(layout);
        deactionRate->setObjectName(QStringLiteral("deactionRate"));
        deactionRate->setDecimals(5);
        deactionRate->setValue(0.1);

        gridLayout->addWidget(deactionRate, 3, 1, 1, 1);


        retranslateUi(AlgaePropertyWidget);

        QMetaObject::connectSlotsByName(AlgaePropertyWidget);
    } // setupUi

    void retranslateUi(QWidget *AlgaePropertyWidget)
    {
        AlgaePropertyWidget->setWindowTitle(QApplication::translate("AlgaePropertyWidget", "Form", 0));
        reactionRateLabel->setText(QApplication::translate("AlgaePropertyWidget", "Reaction Rate", 0));
        bioThresholdLabel->setText(QApplication::translate("AlgaePropertyWidget", "Bioluminescence Threshold", 0));
        deactionRateLabel->setText(QApplication::translate("AlgaePropertyWidget", "Deaction Rate", 0));
    } // retranslateUi

};

namespace Ui {
    class AlgaePropertyWidget: public Ui_AlgaePropertyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ALGAEPROPERTYWIDGET_H
