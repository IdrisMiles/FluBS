/********************************************************************************
** Form generated from reading UI file 'rigidpropertywidget.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_RIGIDPROPERTYWIDGET_H
#define UI_RIGIDPROPERTYWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_RigidPropertyWidget
{
public:
    QWidget *layout;
    QGridLayout *gridLayout;
    QLabel *label_2;
    QSpacerItem *verticalSpacer;
    QDoubleSpinBox *rotY;
    QCheckBox *static_2;
    QFrame *line_2;
    QCheckBox *kinematic;
    QDoubleSpinBox *posZ;
    QDoubleSpinBox *posY;
    QDoubleSpinBox *rotZ;
    QDoubleSpinBox *rotX;
    QLabel *label;
    QDoubleSpinBox *posX;
    QFrame *line;
    QLabel *label_3;
    QDoubleSpinBox *scaleX;
    QDoubleSpinBox *scaleY;
    QDoubleSpinBox *scaleZ;

    void setupUi(QWidget *RigidPropertyWidget)
    {
        if (RigidPropertyWidget->objectName().isEmpty())
            RigidPropertyWidget->setObjectName(QStringLiteral("RigidPropertyWidget"));
        RigidPropertyWidget->resize(400, 300);
        layout = new QWidget(RigidPropertyWidget);
        layout->setObjectName(QStringLiteral("layout"));
        layout->setGeometry(QRect(70, 70, 291, 141));
        gridLayout = new QGridLayout(layout);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_2 = new QLabel(layout);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 5, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 7, 0, 1, 1);

        rotY = new QDoubleSpinBox(layout);
        rotY->setObjectName(QStringLiteral("rotY"));
        rotY->setDecimals(4);
        rotY->setMinimum(-360);
        rotY->setMaximum(360);

        gridLayout->addWidget(rotY, 5, 2, 1, 1);

        static_2 = new QCheckBox(layout);
        static_2->setObjectName(QStringLiteral("static_2"));

        gridLayout->addWidget(static_2, 1, 0, 1, 1);

        line_2 = new QFrame(layout);
        line_2->setObjectName(QStringLiteral("line_2"));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line_2, 3, 0, 1, 4);

        kinematic = new QCheckBox(layout);
        kinematic->setObjectName(QStringLiteral("kinematic"));

        gridLayout->addWidget(kinematic, 2, 0, 1, 1);

        posZ = new QDoubleSpinBox(layout);
        posZ->setObjectName(QStringLiteral("posZ"));
        posZ->setDecimals(4);
        posZ->setMinimum(-100);
        posZ->setMaximum(100);

        gridLayout->addWidget(posZ, 4, 3, 1, 1);

        posY = new QDoubleSpinBox(layout);
        posY->setObjectName(QStringLiteral("posY"));
        posY->setDecimals(4);
        posY->setMinimum(-100);
        posY->setMaximum(100);

        gridLayout->addWidget(posY, 4, 2, 1, 1);

        rotZ = new QDoubleSpinBox(layout);
        rotZ->setObjectName(QStringLiteral("rotZ"));
        rotZ->setDecimals(4);
        rotZ->setMinimum(-360);
        rotZ->setMaximum(360);

        gridLayout->addWidget(rotZ, 5, 3, 1, 1);

        rotX = new QDoubleSpinBox(layout);
        rotX->setObjectName(QStringLiteral("rotX"));
        rotX->setDecimals(4);
        rotX->setMinimum(-360);
        rotX->setMaximum(360);

        gridLayout->addWidget(rotX, 5, 1, 1, 1);

        label = new QLabel(layout);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 4, 0, 1, 1);

        posX = new QDoubleSpinBox(layout);
        posX->setObjectName(QStringLiteral("posX"));
        posX->setDecimals(4);
        posX->setMinimum(-100);
        posX->setMaximum(100);

        gridLayout->addWidget(posX, 4, 1, 1, 1);

        line = new QFrame(layout);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line, 0, 0, 1, 4);

        label_3 = new QLabel(layout);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 6, 0, 1, 1);

        scaleX = new QDoubleSpinBox(layout);
        scaleX->setObjectName(QStringLiteral("scaleX"));
        scaleX->setDecimals(4);
        scaleX->setMinimum(0.0001);
        scaleX->setValue(1);

        gridLayout->addWidget(scaleX, 6, 1, 1, 1);

        scaleY = new QDoubleSpinBox(layout);
        scaleY->setObjectName(QStringLiteral("scaleY"));
        scaleY->setDecimals(4);
        scaleY->setMinimum(0.0001);
        scaleY->setValue(1);

        gridLayout->addWidget(scaleY, 6, 2, 1, 1);

        scaleZ = new QDoubleSpinBox(layout);
        scaleZ->setObjectName(QStringLiteral("scaleZ"));
        scaleZ->setDecimals(4);
        scaleZ->setMinimum(0.0001);
        scaleZ->setValue(1);

        gridLayout->addWidget(scaleZ, 6, 3, 1, 1);


        retranslateUi(RigidPropertyWidget);

        QMetaObject::connectSlotsByName(RigidPropertyWidget);
    } // setupUi

    void retranslateUi(QWidget *RigidPropertyWidget)
    {
        RigidPropertyWidget->setWindowTitle(QApplication::translate("RigidPropertyWidget", "Form", 0));
        label_2->setText(QApplication::translate("RigidPropertyWidget", "Rotation", 0));
        static_2->setText(QApplication::translate("RigidPropertyWidget", "Static", 0));
        kinematic->setText(QApplication::translate("RigidPropertyWidget", "Kinematic", 0));
        label->setText(QApplication::translate("RigidPropertyWidget", "Position", 0));
        label_3->setText(QApplication::translate("RigidPropertyWidget", "Scale", 0));
    } // retranslateUi

};

namespace Ui {
    class RigidPropertyWidget: public Ui_RigidPropertyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_RIGIDPROPERTYWIDGET_H
