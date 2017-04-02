#ifndef TIMELINEWIDGET_H
#define TIMELINEWIDGET_H

#include <QFrame>
#include <memory>
#include <Widget/timeline.h>

namespace Ui {
class TimeLineWidget;
}

class TimeLineWidget : public QFrame
{
    Q_OBJECT

public:
    explicit TimeLineWidget(QWidget *parent = 0);
    ~TimeLineWidget();

public slots:
    void OnFrameChanged(int frame);

signals:
    void FrameChanged(int);

protected:

private:
    Ui::TimeLineWidget *ui;
    std::unique_ptr<TimeLine> m_timeLine;

};

#endif // TIMELINEWIDGET_H
