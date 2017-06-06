#ifndef TIMELINEWIDGET_H
#define TIMELINEWIDGET_H

#include <QFrame>
#include <memory>
#include <Widget/timeline.h>

//--------------------------------------------------------------------------------------------------------------
/// @author Idris Miles
/// @version 1.0
/// @date 01/06/2017
//--------------------------------------------------------------------------------------------------------------


namespace Ui {
class TimeLineWidget;
}

/// @class TimeLineWidget
/// @brief Inherits from QFrame. Custom timeline widget.
class TimeLineWidget : public QFrame
{
    Q_OBJECT

public:
    /// @brief constructor
    explicit TimeLineWidget(QWidget *parent = 0);

    /// @brief destructor
    ~TimeLineWidget();

    /// @brief Method to pause timeline
    void Pause();

    /// @brief Method to play timeline
    void Play();

public slots:

    /// @brief Slot to handle frame change
    void OnFrameChanged(int frame);

    /// @brief slot to handle frame cached
    void OnFrameCached(int frame);

    /// @brief slot to handle frame being finished
    void OnFrameFinished(int frame);

    /// @brief slot to handle frame range change
    void OnFrameRangeChanged(int frameRange);

signals:
    /// @brief Qt Signal to communicate a frame change
    void FrameChanged(int frame);

    /// @brief Qt Signal to communicate a frame being cahced
    void FrameCached(int frame);

    /// @brief Qt Signal to communicate the cache checkbox state
    void CacheChecked(bool checked);

protected:

private:
    Ui::TimeLineWidget *ui;
    std::unique_ptr<TimeLine> m_timeLine;

};

#endif // TIMELINEWIDGET_H
