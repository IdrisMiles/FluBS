#ifndef TIMELINE_H
#define TIMELINE_H

#include <QTimeLine>

class TimeLine : public QTimeLine
{
    Q_OBJECT

public:

    enum CacheState
    {
        NotCached,
        Cached,
        StaleCache
    };

    TimeLine(int duration = 1000, QObject *parent = nullptr);
    ~TimeLine();

    void SetFrameRange(int start, int end);


protected:

private:
    std::vector<CacheState> frameCacheStates;

};

#endif // TIMELINE_H
