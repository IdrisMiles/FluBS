#include "testtimeline.h"
#include "ui_testtimeline.h"
#include <QStyle>

TestTimeLine::TestTimeLine(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::TestTimeLine)
{
    ui->setupUi(this);
    ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    ui->pauseButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    ui->gotoStartButton->setIcon(style()->standardIcon(QStyle::SP_MediaSkipBackward));
    ui->gotoEndButton->setIcon(style()->standardIcon(QStyle::SP_MediaSkipForward));
}

TestTimeLine::~TestTimeLine()
{
    delete ui;
}
