#include "Widget/timelinewidget.h"
#include "ui_timelinewidget.h"
#include <QStyle>

TimeLineWidget::TimeLineWidget(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::TimeLineWidget)
{
    ui->setupUi(this);
    ui->playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    ui->pauseButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    ui->gotoStartButton->setIcon(style()->standardIcon(QStyle::SP_MediaSkipBackward));
    ui->gotoEndButton->setIcon(style()->standardIcon(QStyle::SP_MediaSkipForward));
    ui->scrubber->setRange(ui->startFrame->value(), ui->endFrame->value());

    // initialise timeline
    m_timeLine = std::unique_ptr<TimeLine>(new TimeLine((ui->endFrame->value() - ui->startFrame->value())* ui->fps->value() * 1000, this));

    // initialise default values
    m_timeLine->setLoopCount(0);
    m_timeLine->SetFrameRange(ui->startFrame->value(), ui->endFrame->value());
    m_timeLine->setCurveShape(TimeLine::CurveShape::LinearCurve);
    m_timeLine->setDuration(1000 * (ui->endFrame->value() - ui->startFrame->value())/ ui->fps->value());



    //------------------------------
    // Setup connections
    connect(ui->startFrame, QOverload<int>::of(&QSpinBox::valueChanged), [this](int value){
        m_timeLine->setStartFrame(value);
        m_timeLine->setDuration(1000 * (ui->endFrame->value() - ui->startFrame->value())/ ui->fps->value());
        ui->scrubber->setRange(value, ui->endFrame->value());
        ui->scrubber->setValue(m_timeLine->currentFrame());
    });

    connect(ui->endFrame, QOverload<int>::of(&QSpinBox::valueChanged), [this](int value){
        m_timeLine->setEndFrame(value);
        m_timeLine->setDuration(1000 * (ui->endFrame->value() - ui->startFrame->value())/ ui->fps->value());
        ui->scrubber->setRange(ui->startFrame->value(), value);
        ui->scrubber->setValue(m_timeLine->currentFrame());
    });

    connect(ui->fps, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double value){
        m_timeLine->setUpdateInterval(1000/value);
        m_timeLine->setDuration(1000 * (ui->endFrame->value() - ui->startFrame->value())/ ui->fps->value());
    });

    connect(ui->playButton, &QPushButton::clicked, [this](bool checked){
        Play();
    });

    connect(ui->pauseButton, &QPushButton::clicked, [this](bool checked){
        Pause();
    });

    connect(ui->gotoStartButton, &QPushButton::clicked, [this](bool checked){
        m_timeLine->setCurrentTime(0);
    });

    connect(ui->gotoEndButton, &QPushButton::clicked, [this](bool checked){
        m_timeLine->setCurrentTime((1000*ui->endFrame->value()-1)/ui->fps->value());
    });

    connect(ui->scrubber, &QSlider::sliderMoved,[this](int frame){
        if(m_timeLine->state() != TimeLine::State::Running)
        {
            m_timeLine->setCurrentTime((1000*frame)/ui->fps->value());
        }
    });

    connect(ui->frame, QOverload<int>::of(&QSpinBox::valueChanged), [this](int value){
        if(m_timeLine->state() != TimeLine::State::Running)
        {
            m_timeLine->setCurrentTime((1000*value)/ui->fps->value());
        }
    });

    connect(m_timeLine.get(), &TimeLine::frameChanged, ui->scrubber, &QSlider::setValue);
    connect(m_timeLine.get(), &TimeLine::frameChanged, ui->frame, &QSpinBox::setValue);
    connect(m_timeLine.get(), &TimeLine::frameChanged, this, &TimeLineWidget::OnFrameChanged);


    connect(ui->cache, &QCheckBox::clicked, [this](bool checked){
        emit CacheChecked(checked);
    });

}

TimeLineWidget::~TimeLineWidget()
{
    delete ui;
}

void TimeLineWidget::Pause()
{
//    m_timeLine->SetSavedState(TimeLine::State::Paused);

    auto state = m_timeLine->state();
    if(state == TimeLine::State::Running)
    {
        m_timeLine->setPaused(true);
    }
}

void TimeLineWidget::Play()
{
//    m_timeLine->SetSavedState(TimeLine::State::Running);

    auto state = m_timeLine->state();
    if(state == TimeLine::State::NotRunning)
    {
        m_timeLine->resume();
    }
    else if(state == TimeLine::State::Paused)
    {
        m_timeLine->resume();
    }
}

void TimeLineWidget::OnFrameChanged(int frame)
{
    emit FrameChanged(frame);
}


void TimeLineWidget::OnFrameCached(int frame)
{

}

void TimeLineWidget::OnFrameFinished(int frame)
{
    m_timeLine->OnFrameFinished(frame);
}
