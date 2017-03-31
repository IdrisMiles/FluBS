#ifndef TIMELINE_H
#define TIMELINE_H

#include <QTimeLine>

class TimeLine : public QTimeLine
{
public:
    Q_OBJECT
    TimeLine(int duration = 1000, QObject *parent = nullptr);
    ~TimeLine();

private:

};

#endif // TIMELINE_H
