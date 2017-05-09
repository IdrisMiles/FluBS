/********************************************************************************
** Form generated from reading UI file 'timelinewidget.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TIMELINEWIDGET_H
#define UI_TIMELINEWIDGET_H

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
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>

QT_BEGIN_NAMESPACE

class Ui_TimeLineWidget
{
public:
    QGridLayout *gridLayout;
    QLabel *frameLabel;
    QSlider *scrubber;
    QPushButton *playButton;
    QPushButton *gotoEndButton;
    QSpinBox *endFrame;
    QSpinBox *frame;
    QLabel *fpsLabel;
    QPushButton *gotoStartButton;
    QLabel *startLabel;
    QSpinBox *startFrame;
    QPushButton *pauseButton;
    QDoubleSpinBox *fps;
    QLabel *endLabel;
    QCheckBox *cache;

    void setupUi(QFrame *TimeLineWidget)
    {
        if (TimeLineWidget->objectName().isEmpty())
            TimeLineWidget->setObjectName(QStringLiteral("TimeLineWidget"));
        TimeLineWidget->resize(798, 78);
        TimeLineWidget->setFrameShape(QFrame::StyledPanel);
        TimeLineWidget->setFrameShadow(QFrame::Raised);
        gridLayout = new QGridLayout(TimeLineWidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        frameLabel = new QLabel(TimeLineWidget);
        frameLabel->setObjectName(QStringLiteral("frameLabel"));

        gridLayout->addWidget(frameLabel, 2, 3, 1, 1);

        scrubber = new QSlider(TimeLineWidget);
        scrubber->setObjectName(QStringLiteral("scrubber"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(scrubber->sizePolicy().hasHeightForWidth());
        scrubber->setSizePolicy(sizePolicy);
        scrubber->setLayoutDirection(Qt::LeftToRight);
        scrubber->setStyleSheet(QStringLiteral(""));
        scrubber->setOrientation(Qt::Horizontal);
        scrubber->setTickPosition(QSlider::TicksBothSides);
        scrubber->setTickInterval(1);

        gridLayout->addWidget(scrubber, 0, 2, 3, 1);

        playButton = new QPushButton(TimeLineWidget);
        playButton->setObjectName(QStringLiteral("playButton"));

        gridLayout->addWidget(playButton, 0, 0, 1, 1);

        gotoEndButton = new QPushButton(TimeLineWidget);
        gotoEndButton->setObjectName(QStringLiteral("gotoEndButton"));

        gridLayout->addWidget(gotoEndButton, 2, 1, 1, 1);

        endFrame = new QSpinBox(TimeLineWidget);
        endFrame->setObjectName(QStringLiteral("endFrame"));
        endFrame->setMaximum(1000);
        endFrame->setValue(250);

        gridLayout->addWidget(endFrame, 0, 6, 1, 1);

        frame = new QSpinBox(TimeLineWidget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setMaximum(1000);

        gridLayout->addWidget(frame, 2, 4, 1, 1);

        fpsLabel = new QLabel(TimeLineWidget);
        fpsLabel->setObjectName(QStringLiteral("fpsLabel"));

        gridLayout->addWidget(fpsLabel, 2, 5, 1, 1);

        gotoStartButton = new QPushButton(TimeLineWidget);
        gotoStartButton->setObjectName(QStringLiteral("gotoStartButton"));

        gridLayout->addWidget(gotoStartButton, 2, 0, 1, 1);

        startLabel = new QLabel(TimeLineWidget);
        startLabel->setObjectName(QStringLiteral("startLabel"));

        gridLayout->addWidget(startLabel, 0, 3, 1, 1);

        startFrame = new QSpinBox(TimeLineWidget);
        startFrame->setObjectName(QStringLiteral("startFrame"));
        startFrame->setMaximum(1000);

        gridLayout->addWidget(startFrame, 0, 4, 1, 1);

        pauseButton = new QPushButton(TimeLineWidget);
        pauseButton->setObjectName(QStringLiteral("pauseButton"));

        gridLayout->addWidget(pauseButton, 0, 1, 1, 1);

        fps = new QDoubleSpinBox(TimeLineWidget);
        fps->setObjectName(QStringLiteral("fps"));
        fps->setMaximum(60);
        fps->setValue(25);

        gridLayout->addWidget(fps, 2, 6, 1, 1);

        endLabel = new QLabel(TimeLineWidget);
        endLabel->setObjectName(QStringLiteral("endLabel"));

        gridLayout->addWidget(endLabel, 0, 5, 1, 1);

        cache = new QCheckBox(TimeLineWidget);
        cache->setObjectName(QStringLiteral("cache"));
        cache->setChecked(true);

        gridLayout->addWidget(cache, 2, 7, 1, 1);


        retranslateUi(TimeLineWidget);

        QMetaObject::connectSlotsByName(TimeLineWidget);
    } // setupUi

    void retranslateUi(QFrame *TimeLineWidget)
    {
        TimeLineWidget->setWindowTitle(QApplication::translate("TimeLineWidget", "Frame", 0));
        frameLabel->setText(QApplication::translate("TimeLineWidget", "Frame", 0));
        playButton->setText(QString());
        gotoEndButton->setText(QString());
        fpsLabel->setText(QApplication::translate("TimeLineWidget", "FPS", 0));
        gotoStartButton->setText(QString());
        startLabel->setText(QApplication::translate("TimeLineWidget", "Start", 0));
        pauseButton->setText(QString());
        endLabel->setText(QApplication::translate("TimeLineWidget", "End", 0));
        cache->setText(QApplication::translate("TimeLineWidget", "Cache", 0));
    } // retranslateUi

};

namespace Ui {
    class TimeLineWidget: public Ui_TimeLineWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TIMELINEWIDGET_H
