/********************************************************************************
** Form generated from reading UI file 'solverpropertywidget.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SOLVERPROPERTYWIDGET_H
#define UI_SOLVERPROPERTYWIDGET_H

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

class Ui_SolverPropertyWidget
{
public:
    QGridLayout *gridLayout;
    QLabel *deltaTimeLabel;
    QLabel *gridCellSizeLabel;
    QLabel *smoothingLengthLabel;
    QLabel *solveIterationsLabel;
    QLabel *gridResLabel;
    QSpinBox *solveIterations;
    QDoubleSpinBox *deltaTime;
    QDoubleSpinBox *smoothingLength;
    QDoubleSpinBox *gridCellSize;
    QSpinBox *gridRes;

    void setupUi(QWidget *SolverPropertyWidget)
    {
        if (SolverPropertyWidget->objectName().isEmpty())
            SolverPropertyWidget->setObjectName(QStringLiteral("SolverPropertyWidget"));
        SolverPropertyWidget->resize(400, 300);
        gridLayout = new QGridLayout(SolverPropertyWidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        deltaTimeLabel = new QLabel(SolverPropertyWidget);
        deltaTimeLabel->setObjectName(QStringLiteral("deltaTimeLabel"));

        gridLayout->addWidget(deltaTimeLabel, 3, 0, 1, 1);

        gridCellSizeLabel = new QLabel(SolverPropertyWidget);
        gridCellSizeLabel->setObjectName(QStringLiteral("gridCellSizeLabel"));

        gridLayout->addWidget(gridCellSizeLabel, 1, 0, 1, 1);

        smoothingLengthLabel = new QLabel(SolverPropertyWidget);
        smoothingLengthLabel->setObjectName(QStringLiteral("smoothingLengthLabel"));

        gridLayout->addWidget(smoothingLengthLabel, 2, 0, 1, 1);

        solveIterationsLabel = new QLabel(SolverPropertyWidget);
        solveIterationsLabel->setObjectName(QStringLiteral("solveIterationsLabel"));

        gridLayout->addWidget(solveIterationsLabel, 4, 0, 1, 1);

        gridResLabel = new QLabel(SolverPropertyWidget);
        gridResLabel->setObjectName(QStringLiteral("gridResLabel"));

        gridLayout->addWidget(gridResLabel, 0, 0, 1, 1);

        solveIterations = new QSpinBox(SolverPropertyWidget);
        solveIterations->setObjectName(QStringLiteral("solveIterations"));

        gridLayout->addWidget(solveIterations, 4, 1, 1, 1);

        deltaTime = new QDoubleSpinBox(SolverPropertyWidget);
        deltaTime->setObjectName(QStringLiteral("deltaTime"));
        deltaTime->setDecimals(5);
        deltaTime->setMinimum(0.001);
        deltaTime->setMaximum(1);
        deltaTime->setSingleStep(0.0001);
        deltaTime->setValue(0.005);

        gridLayout->addWidget(deltaTime, 3, 1, 1, 1);

        smoothingLength = new QDoubleSpinBox(SolverPropertyWidget);
        smoothingLength->setObjectName(QStringLiteral("smoothingLength"));

        gridLayout->addWidget(smoothingLength, 2, 1, 1, 1);

        gridCellSize = new QDoubleSpinBox(SolverPropertyWidget);
        gridCellSize->setObjectName(QStringLiteral("gridCellSize"));

        gridLayout->addWidget(gridCellSize, 1, 1, 1, 1);

        gridRes = new QSpinBox(SolverPropertyWidget);
        gridRes->setObjectName(QStringLiteral("gridRes"));

        gridLayout->addWidget(gridRes, 0, 1, 1, 1);


        retranslateUi(SolverPropertyWidget);

        QMetaObject::connectSlotsByName(SolverPropertyWidget);
    } // setupUi

    void retranslateUi(QWidget *SolverPropertyWidget)
    {
        SolverPropertyWidget->setWindowTitle(QApplication::translate("SolverPropertyWidget", "Form", 0));
        deltaTimeLabel->setText(QApplication::translate("SolverPropertyWidget", "Delta Time", 0));
        gridCellSizeLabel->setText(QApplication::translate("SolverPropertyWidget", "Grid Cell Size", 0));
        smoothingLengthLabel->setText(QApplication::translate("SolverPropertyWidget", "Smoothing Length", 0));
        solveIterationsLabel->setText(QApplication::translate("SolverPropertyWidget", "Solve Iterations", 0));
        gridResLabel->setText(QApplication::translate("SolverPropertyWidget", "Grid Resolution", 0));
    } // retranslateUi

};

namespace Ui {
    class SolverPropertyWidget: public Ui_SolverPropertyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SOLVERPROPERTYWIDGET_H
