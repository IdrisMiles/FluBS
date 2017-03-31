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
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_RigidPropertyWidget
{
public:
    QWidget *layout;
    QGridLayout *gridLayout;
    QFrame *line;
    QCheckBox *kinematic;
    QCheckBox *static_2;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *RigidPropertyWidget)
    {
        if (RigidPropertyWidget->objectName().isEmpty())
            RigidPropertyWidget->setObjectName(QStringLiteral("RigidPropertyWidget"));
        RigidPropertyWidget->resize(400, 300);
        layout = new QWidget(RigidPropertyWidget);
        layout->setObjectName(QStringLiteral("layout"));
        layout->setGeometry(QRect(70, 70, 103, 64));
        gridLayout = new QGridLayout(layout);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        line = new QFrame(layout);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line, 0, 0, 1, 1);

        kinematic = new QCheckBox(layout);
        kinematic->setObjectName(QStringLiteral("kinematic"));

        gridLayout->addWidget(kinematic, 2, 0, 1, 1);

        static_2 = new QCheckBox(layout);
        static_2->setObjectName(QStringLiteral("static_2"));

        gridLayout->addWidget(static_2, 1, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 3, 0, 1, 1);


        retranslateUi(RigidPropertyWidget);

        QMetaObject::connectSlotsByName(RigidPropertyWidget);
    } // setupUi

    void retranslateUi(QWidget *RigidPropertyWidget)
    {
        RigidPropertyWidget->setWindowTitle(QApplication::translate("RigidPropertyWidget", "Form", 0));
        kinematic->setText(QApplication::translate("RigidPropertyWidget", "Kinematic", 0));
        static_2->setText(QApplication::translate("RigidPropertyWidget", "Static", 0));
    } // retranslateUi

};

namespace Ui {
    class RigidPropertyWidget: public Ui_RigidPropertyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_RIGIDPROPERTYWIDGET_H
