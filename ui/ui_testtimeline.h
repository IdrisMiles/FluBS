/********************************************************************************
** Form generated from reading UI file 'testtimeline.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TESTTIMELINE_H
#define UI_TESTTIMELINE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>

QT_BEGIN_NAMESPACE

class Ui_TestTimeLine
{
public:
    QGridLayout *gridLayout;
    QLabel *label_3;
    QLabel *label;
    QSpinBox *spinBox_2;
    QPushButton *gotoStartButton;
    QSlider *horizontalSlider;
    QPushButton *playButton;
    QPushButton *pauseButton;
    QPushButton *gotoEndButton;
    QDoubleSpinBox *doubleSpinBox;
    QSpinBox *spinBox;
    QLabel *label_2;

    void setupUi(QFrame *TestTimeLine)
    {
        if (TestTimeLine->objectName().isEmpty())
            TestTimeLine->setObjectName(QStringLiteral("TestTimeLine"));
        TestTimeLine->resize(798, 69);
        TestTimeLine->setFrameShape(QFrame::StyledPanel);
        TestTimeLine->setFrameShadow(QFrame::Raised);
        gridLayout = new QGridLayout(TestTimeLine);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_3 = new QLabel(TestTimeLine);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 2, 3, 1, 1);

        label = new QLabel(TestTimeLine);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 0, 3, 1, 1);

        spinBox_2 = new QSpinBox(TestTimeLine);
        spinBox_2->setObjectName(QStringLiteral("spinBox_2"));

        gridLayout->addWidget(spinBox_2, 0, 6, 1, 1);

        gotoStartButton = new QPushButton(TestTimeLine);
        gotoStartButton->setObjectName(QStringLiteral("gotoStartButton"));

        gridLayout->addWidget(gotoStartButton, 2, 0, 1, 1);

        horizontalSlider = new QSlider(TestTimeLine);
        horizontalSlider->setObjectName(QStringLiteral("horizontalSlider"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(horizontalSlider->sizePolicy().hasHeightForWidth());
        horizontalSlider->setSizePolicy(sizePolicy);
        horizontalSlider->setLayoutDirection(Qt::LeftToRight);
        horizontalSlider->setOrientation(Qt::Horizontal);
        horizontalSlider->setTickPosition(QSlider::TicksBothSides);
        horizontalSlider->setTickInterval(1);

        gridLayout->addWidget(horizontalSlider, 0, 2, 3, 1);

        playButton = new QPushButton(TestTimeLine);
        playButton->setObjectName(QStringLiteral("playButton"));

        gridLayout->addWidget(playButton, 0, 0, 1, 1);

        pauseButton = new QPushButton(TestTimeLine);
        pauseButton->setObjectName(QStringLiteral("pauseButton"));

        gridLayout->addWidget(pauseButton, 0, 1, 1, 1);

        gotoEndButton = new QPushButton(TestTimeLine);
        gotoEndButton->setObjectName(QStringLiteral("gotoEndButton"));

        gridLayout->addWidget(gotoEndButton, 2, 1, 1, 1);

        doubleSpinBox = new QDoubleSpinBox(TestTimeLine);
        doubleSpinBox->setObjectName(QStringLiteral("doubleSpinBox"));

        gridLayout->addWidget(doubleSpinBox, 2, 4, 1, 1);

        spinBox = new QSpinBox(TestTimeLine);
        spinBox->setObjectName(QStringLiteral("spinBox"));

        gridLayout->addWidget(spinBox, 0, 4, 1, 1);

        label_2 = new QLabel(TestTimeLine);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 0, 5, 1, 1);


        retranslateUi(TestTimeLine);

        QMetaObject::connectSlotsByName(TestTimeLine);
    } // setupUi

    void retranslateUi(QFrame *TestTimeLine)
    {
        TestTimeLine->setWindowTitle(QApplication::translate("TestTimeLine", "Frame", 0));
        label_3->setText(QApplication::translate("TestTimeLine", "FPS", 0));
        label->setText(QApplication::translate("TestTimeLine", "Start", 0));
        gotoStartButton->setText(QString());
        playButton->setText(QString());
        pauseButton->setText(QString());
        gotoEndButton->setText(QString());
        label_2->setText(QApplication::translate("TestTimeLine", "End", 0));
    } // retranslateUi

};

namespace Ui {
    class TestTimeLine: public Ui_TestTimeLine {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TESTTIMELINE_H
